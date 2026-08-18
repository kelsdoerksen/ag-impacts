"""
Model registry for yield prediction.
Instantiate with a model type string; the class wraps fit/predict/score
and exposes feature_importances_ when available.

Supported types:
    rf  — RandomForestRegressor
    gbm — GradientBoostingRegressor
    xgb — XGBRegressor (requires xgboost package)
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

SUPPORTED_MODELS = ["rf", "gbm", "xgb"]

# Models that accept NaN in X and learn a default split direction for it.
# sklearn's RandomForestRegressor and GradientBoostingRegressor both reject
# NaN, so feature blocks with legitimate gaps (e.g. per-month precipitation for
# a month outside a county's crop calendar) can only be kept for these.
NAN_TOLERANT_MODELS = ["xgb"]


class YieldModel:
    def __init__(self, model_type: str, random_state: int = 42):
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from: {SUPPORTED_MODELS}"
            )
        self.model_type = model_type
        self._model = self._build(model_type, random_state)

    def _build(self, model_type, random_state):
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

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def score(self, X, y):
        return self._model.score(X, y)

    @property
    def feature_importances_(self):
        return getattr(self._model, "feature_importances_", None)
