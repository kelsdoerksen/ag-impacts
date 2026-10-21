"""
Model registry for yield prediction.
Instantiate with a model type string; the class wraps fit/predict/score
and exposes feature_importances_ when available.

Supported types:
    rf    — RandomForestRegressor
    gbm   — GradientBoostingRegressor
    xgb   — XGBRegressor (requires xgboost package)
    ridge — RidgeCV, standardised (linear, heavily regularised)
    enet  — ElasticNetCV, standardised (linear, sparse + regularised)

WHY LINEAR MODELS BELONG HERE

The ensembles calibrate their flexibility to the ROW count, and for seasonal
climate features that count is misleading. The part of yield that weather moves
is largely a common year effect -- districts within a year share their weather,
so their residuals are correlated (r ~ 0.28 for Zambian maize). The number of
independent realisations is therefore closer to the number of YEARS (tens) than
the number of district-years (thousands). A 200-tree forest against ~33
effective points fits idiosyncrasy and carries it into the held-out year as
noise, which shows up as a strongly negative anomaly R2 next to a healthy
absolute R2.

Ridge is the natural counterpart: its capacity is set by a penalty chosen on the
data rather than by the row count, and where there is no signal it shrinks
towards the mean instead of inventing one. Both linear models standardise first,
inside the pipeline, so the penalty is applied on comparable scales and the fold
statistics are learned on training rows only.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SUPPORTED_MODELS = ["rf", "gbm", "xgb", "ridge", "enet", "mlp", "logreg"]

# Which models can do which job. ridge/enet are least-squares regressors and
# logreg is a classifier; the rest do both.
TASKS = ["regress", "classify"]
REGRESSORS  = ["rf", "gbm", "xgb", "ridge", "enet", "mlp"]
CLASSIFIERS = ["rf", "gbm", "xgb", "mlp", "logreg"]

# Weight decay for the MLP, chosen by grouped CV like the ridge penalty. The
# useful value spans orders of magnitude for the same reason it does there, but
# the grid is deliberately coarse: every extra point costs n_splits network
# fits inside every outer fold, and the selection only has to find the right
# order of magnitude. Four points over six decades does that.
MLP_ALPHAS = np.logspace(-3, 3, 4)

# Spanning ten orders of magnitude: the useful penalty is tiny when the features
# genuinely carry the signal and enormous when they do not, and which regime a
# given AOI/feature-space lands in is exactly what these runs are measuring.
RIDGE_ALPHAS = np.logspace(-3, 7, 41)


class GroupedRidgeCV(RegressorMixin, BaseEstimator):
    """Ridge whose penalty is chosen by grouped CV, falling back to row-wise.

    Inherits BaseEstimator so Pipeline can introspect it: from scikit-learn 1.6
    `check_is_fitted` goes through `get_tags`, which needs `__sklearn_tags__`,
    and a plain class raises AttributeError at predict time rather than at fit.
    RegressorMixin goes first so BaseEstimator sits rightmost in the MRO, as
    sklearn requires.

    Why not sklearn's RidgeCV directly: its default generalised CV holds out one
    ROW at a time, which assumes rows are independent. Seasonal climate features
    are shared across the districts of a country in a given year, so holding out
    one district-year leaves 70 near-duplicates of it in the training set. The CV
    error therefore looks far better than it is at every alpha, and the selected
    penalty comes out much too small -- the same effective-sample-size mistake
    the forest makes, now inside the regularisation that was supposed to fix it.

    Measured on a synthetic panel with a target unrelated to the features,
    row-wise GCV picked alpha=178 and left coefficients up to 0.28; holding out
    whole years drives alpha to the top of the grid and the coefficients to zero,
    which is the honest answer.

    So when `groups` are supplied at fit time (the training rows' YEARS), alpha
    is chosen by holding out whole groups. Without groups it falls back to
    RidgeCV, which is correct for genuinely independent rows.
    """

    def __init__(self, alphas=RIDGE_ALPHAS, n_splits: int = 5):
        # Stored verbatim, not coerced: sklearn's get_params/clone contract is
        # that __init__ records its arguments untouched.
        self.alphas = alphas
        self.n_splits = n_splits

    def fit(self, X, y, groups=None):
        from sklearn.linear_model import Ridge
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        alphas = np.asarray(self.alphas, dtype=float)

        n_groups = 0 if groups is None else len(np.unique(groups))
        if groups is None or n_groups < 3:
            base = RidgeCV(alphas=alphas).fit(X, y)
            self.alpha_ = float(base.alpha_)
            self.n_groups_ = n_groups
        else:
            from sklearn.model_selection import GroupKFold
            splits = list(GroupKFold(n_splits=min(self.n_splits, n_groups)).split(X, y, groups))
            err = np.zeros(len(alphas))
            for tr, te in splits:
                for i, a in enumerate(alphas):
                    r = Ridge(alpha=a).fit(X[tr], y[tr])
                    err[i] += float(((y[te] - r.predict(X[te])) ** 2).sum())
            self.alpha_ = float(alphas[int(np.argmin(err))])
            self.n_groups_ = n_groups
            self.cv_curve_ = err / len(y)

        self._est = Ridge(alpha=self.alpha_).fit(X, y)
        self.coef_ = self._est.coef_
        self.intercept_ = self._est.intercept_
        return self

    def predict(self, X):
        return self._est.predict(np.asarray(X, dtype=float))

    def score(self, X, y):
        return self._est.score(np.asarray(X, dtype=float), np.asarray(y, dtype=float))


class GroupedMLP(BaseEstimator):
    """An MLP whose weight decay is chosen by holding out whole GROUPS.

    Two deliberate departures from sklearn's defaults.

    No early stopping. MLPRegressor's early_stopping holds out a random
    FRACTION OF ROWS, and for a climate panel those rows are not independent:
    the districts of a country share a year's weather, so a random split leaves
    near-duplicates of each validation row in the training half. The validation
    curve then keeps improving long after the model has started fitting
    year-specific idiosyncrasy, and training stops too late. The same
    effective-sample-size mistake GroupedRidgeCV exists to avoid.

    Capacity is set by `alpha` instead, selected by GroupKFold over the
    training rows' YEARS. Where there is no signal the penalty goes to the top
    of the grid and the network collapses towards the mean, which is the honest
    answer; where there is signal it stays small.

    WHY AN MLP AT ALL, next to a forest. A forest predicts an average of leaf
    means, so its output is bounded by the range of the training targets and
    the anomalies it implies are bounded by anomalies it has already seen. An
    MLP's output layer is affine and extrapolates, so it can predict a
    deviation larger than any in training. That is the difference that shows up
    in sd_ratio_anom -- not that one model is better, but that one of them is
    structurally incapable of an unprecedented year.
    """

    def __init__(self, task: str = "regress", hidden=(64, 32),
                 alphas=MLP_ALPHAS, n_splits: int = 3, max_iter: int = 500,
                 random_state: int = 42):
        self.task = task
        self.hidden = hidden
        self.alphas = alphas
        self.n_splits = n_splits
        self.max_iter = max_iter
        self.random_state = random_state

    def _net(self, alpha):
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        cls = MLPClassifier if self.task == "classify" else MLPRegressor
        return cls(hidden_layer_sizes=tuple(self.hidden), alpha=float(alpha),
                   activation="relu", solver="adam", learning_rate_init=1e-3,
                   max_iter=int(self.max_iter), early_stopping=False,
                   n_iter_no_change=15, shuffle=True,
                   random_state=self.random_state)

    def fit(self, X, y, groups=None):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.task == "regress":
            y = y.astype(float)
        alphas = np.asarray(self.alphas, dtype=float)
        n_groups = 0 if groups is None else len(np.unique(groups))
        self.n_groups_ = n_groups

        with warnings.catch_warnings():
            # adam on a few thousand rows often stops on max_iter rather than
            # tol; that is a tuning signal, not a failure, and one warning per
            # alpha per fold would bury the run log.
            warnings.simplefilter("ignore", ConvergenceWarning)
            if n_groups < 3 or len(alphas) == 1:
                self.alpha_ = float(alphas[len(alphas) // 2])
            else:
                from sklearn.model_selection import GroupKFold
                splits = list(GroupKFold(n_splits=min(self.n_splits, n_groups))
                              .split(X, y, groups))
                err = np.zeros(len(alphas))
                for tr, te in splits:
                    for i, a in enumerate(alphas):
                        m = self._net(a).fit(X[tr], y[tr])
                        if self.task == "classify":
                            err[i] += float((m.predict(X[te]) != y[te]).sum())
                        else:
                            err[i] += float(((y[te] - m.predict(X[te])) ** 2).sum())
                self.alpha_ = float(alphas[int(np.argmin(err))])
                self.cv_curve_ = err / len(y)
            self._est = self._net(self.alpha_).fit(X, y)
        if self.task == "classify":
            self.classes_ = self._est.classes_
        return self

    def predict(self, X):
        return self._est.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X):
        return self._est.predict_proba(np.asarray(X, dtype=float))

    def score(self, X, y):
        return self._est.score(np.asarray(X, dtype=float), np.asarray(y))

    def __sklearn_tags__(self):
        # Mirrors what RegressorMixin/ClassifierMixin would set, but the
        # estimator type is only known once `task` is, so it is set here.
        tags = super().__sklearn_tags__()
        tags.estimator_type = ("classifier" if self.task == "classify"
                               else "regressor")
        return tags


class YieldModel:
    """One handle over every estimator, for either task.

    `task` is "regress" (predict kg/ha, or a unit-centred anomaly) or
    "classify" (predict a direction class against a prior year). The same token
    means the same model family in both: `rf` is a random forest either way, so
    a regression run and a direction run differ in the question asked, not in
    the learner.
    """

    def __init__(self, model_type: str, random_state: int = 42,
                 task: str = "regress", mlp_hidden=None, mlp_alpha=None):
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}. Choose from: {TASKS}")
        allowed = CLASSIFIERS if task == "classify" else REGRESSORS
        if model_type not in allowed:
            raise ValueError(
                f"model_type '{model_type}' cannot do task '{task}'. "
                f"Choose from: {allowed}"
            )
        self.model_type = model_type
        self.task = task
        # Pinning alpha skips the grouped search entirely, which is the
        # difference between minutes and hours on a large panel. Worth doing
        # once the search has told you which decade matters.
        self._mlp_kw = {}
        if mlp_hidden:
            self._mlp_kw["hidden"] = tuple(mlp_hidden)
        if mlp_alpha is not None:
            self._mlp_kw["alphas"] = [float(mlp_alpha)]
        self._model = self._build(model_type, random_state, task)

    def _build(self, model_type, random_state, task="regress", **kw):
        if task == "classify":
            return self._build_classifier(model_type, random_state)
        if model_type == "rf":
            return RandomForestRegressor(
                n_estimators=200,
                max_features="sqrt",
                min_samples_leaf=3,
                random_state=random_state,
                n_jobs=-1,
            )
        if model_type == "gbm":
            return GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=3,
                random_state=random_state,
            )
        if model_type == "xgb":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError(
                    "xgboost is not installed. Run: pip install xgboost"
                ) from exc
            return XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
        if model_type == "ridge":
            # Leave-one-out generalised CV over RIDGE_ALPHAS, refit on the whole
            # training fold. The alpha is chosen inside each outer LOO fold, so
            # the held-out year never informs the penalty.
            return Pipeline([
                ("scale", StandardScaler()),
                ("model", GroupedRidgeCV(alphas=RIDGE_ALPHAS)),
            ])
        if model_type == "mlp":
            # Standardisation is not optional for a network: with GDD in the
            # thousands and VPD around one, the first layer's gradients are
            # dominated by whichever feature happens to have the larger unit.
            return Pipeline([
                ("scale", StandardScaler()),
                ("model", GroupedMLP(task="regress", random_state=random_state,
                                     **self._mlp_kw)),
            ])
        if model_type == "enet":
            # l1_ratio grid rather than a fixed value: with correlated seasonal
            # aggregates, pure lasso arbitrarily keeps one of a colinear pair,
            # and the ratio that works is an empirical question per feature set.
            return Pipeline([
                ("scale", StandardScaler()),
                ("model", ElasticNetCV(
                    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                    n_alphas=50, cv=5, random_state=random_state,
                    max_iter=20000, n_jobs=-1)),
            ])

    def _build_classifier(self, model_type, random_state):
        """Classifier counterparts, matched to the regressors above.

        class_weight="balanced" on the tree models and logistic regression:
        with a flat band around zero the three direction classes are usually
        unequal, and an unweighted fit answers "predict the common class" with
        a respectable accuracy and no skill. Weighting makes the model pay for
        missing the rare class, and the reported balanced accuracy and Heidke
        score check that it actually did.
        """
        if model_type == "rf":
            return RandomForestClassifier(
                n_estimators=400, max_features="sqrt", min_samples_leaf=3,
                class_weight="balanced", random_state=random_state, n_jobs=-1)
        if model_type == "gbm":
            # No class_weight on sklearn's boosting; it is reachable only via
            # sample_weight at fit time, which the fold runner does not thread.
            return GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                min_samples_leaf=3, random_state=random_state)
        if model_type == "xgb":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError(
                    "xgboost is not installed. Run: pip install xgboost") from exc
            return XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=random_state,
                n_jobs=-1, verbosity=0, eval_metric="mlogloss")
        if model_type == "mlp":
            return Pipeline([
                ("scale", StandardScaler()),
                ("model", GroupedMLP(task="classify", random_state=random_state,
                                     **self._mlp_kw)),
            ])
        if model_type == "logreg":
            # The regularised linear counterpart, and the one to compare the
            # others against: its penalty is chosen by holding out whole years,
            # so if it wins there was no nonlinearity worth the extra capacity.
            return Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegressionCV(
                    Cs=np.logspace(-4, 3, 15), cv=4, max_iter=5000,
                    class_weight="balanced", scoring="balanced_accuracy",
                    random_state=random_state, n_jobs=-1)),
            ])

    def fit(self, X, y, groups=None):
        """Fit, passing `groups` to estimators that select hyperparameters by CV.

        `groups` should be the training rows' years. Only the linear CV models
        use it -- for the ensembles it is silently ignored, since their capacity
        is set by hyperparameters rather than chosen on the data.
        """
        if groups is not None and self._uses_groups():
            self._model.fit(X, y, model__groups=np.asarray(groups))
        else:
            self._model.fit(X, y)
        return self

    def _uses_groups(self) -> bool:
        if not isinstance(self._model, Pipeline):
            return False
        return isinstance(self._model.named_steps.get("model"),
                          (GroupedRidgeCV, GroupedMLP))

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        """Class probabilities, or None for a regressor / a model without them."""
        fn = getattr(self._model, "predict_proba", None)
        return None if fn is None else fn(X)

    @property
    def classes_(self):
        est = self._model
        if isinstance(est, Pipeline):
            est = est.named_steps.get("model")
        return getattr(est, "classes_", None)

    def score(self, X, y):
        return self._model.score(X, y)

    @property
    def feature_importances_(self):
        """Tree importances, or |standardised coefficient| for the linear models.

        Not the same quantity -- one is impurity reduction, the other is effect
        size on standardised inputs -- but both answer "which columns is this
        model actually using", which is what the callers want. Absolute value
        because the caller ranks them; sign is available via .coef_.
        """
        imp = getattr(self._model, "feature_importances_", None)
        if imp is not None:
            return imp
        coef = self.coef_
        return None if coef is None else np.abs(coef)

    @property
    def coef_(self):
        """Signed standardised coefficients for the linear models, else None.

        The pipeline standardises first, so these are already on a common scale:
        yield units per one standard deviation of each feature.
        """
        est = self._model
        if isinstance(est, Pipeline):
            est = est.named_steps.get("model")
        return getattr(est, "coef_", None)

    @property
    def n_iter_(self):
        """Iterations the network actually ran, or None. Equal to max_iter means
        it was still improving when it was cut off."""
        est = self._model
        if isinstance(est, Pipeline):
            est = est.named_steps.get("model")
        inner = getattr(est, "_est", None)
        return getattr(inner, "n_iter_", None)

    @property
    def alpha_(self):
        """The penalty the CV actually selected, or None. Worth logging: an alpha
        pinned at the top of RIDGE_ALPHAS means the fold found no signal and
        shrank everything to the mean."""
        est = self._model
        if isinstance(est, Pipeline):
            est = est.named_steps.get("model")
        return getattr(est, "alpha_", None)
