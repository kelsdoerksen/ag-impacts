#!/usr/bin/env bash
# Process the EC-Earth3 DCPP hindcasts into per-pixel cubes + admin-2 features,
# for several AOIs, several ensemble members and several lead times.
#
# Stage 1 (R/dcpp-pixel-store.R) reads the daily NetCDFs off the Synology share
# and writes a per-pixel daily cube plus the cropland weight table. Stage 2
# (python/era5_features.py) computes per-pixel daily features. Both are the same
# code the ERA5 path uses, which is the point: a hindcast-vs-reanalysis
# comparison should measure forecast skill, not a difference in processing.
#
# TARGET YEARS vs INITIALISATION YEARS -- the thing to get right.
# A DCPP file for initialisation year s at lead F covers 1 Nov (s+F-1) to
# 31 Oct (s+F), so it verifies against harvest year s+F. Ask for the harvest
# years you want and this works backwards:
#
#     --target-years 2002-2022 --forecast-years 1     ->  s2001..s2021
#     --target-years 2002-2022 --forecast-years 1,2   ->  s2001..s2021 and
#                                                         s2000..s2020
#
# Pass --init-years instead to name initialisation years directly.
#
#   ./run-dcpp.sh --aois ZM,ET,BF                       # plan only
#   ./run-dcpp.sh --aois ZM,ET,BF --go
#   ./run-dcpp.sh --aois ZM --forecast-years 1,2,3,4,5 --go
#   ./run-dcpp.sh --aois ZM --members r1i4p1f1,r2i4p1f1 --go
#
# Each member gets its own output tree, tagged EC-Earth3_{member}: the cube
# filename carries no member token, so members sharing a tree would overwrite
# one another. Pass the same tag to era5_features.py --source later.
#
# Resumable: stage 1 skips any year whose cube already exists, stage 2 likewise,
# and a failure is logged and stepped over rather than stopping the queue.

set -uo pipefail

# ── where things live ────────────────────────────────────────────────────────
SHARE_MOUNT=${SHARE_MOUNT:-/mnt/crl_storage}
# AGIMPACTS_ROOT must be the directory holding cy-bench/ and the cropland
# raster, since R derives every input path from it.
DATA_ROOT=${AGIMPACTS_ROOT:-$SHARE_MOUNT/personal_folders/kdoerksen}
# Root of the published DCPP archive: {DCPP_ROOT}/{member}/day/{var}/gr/{version}
DCPP_ROOT=${DCPP_ROOT:-$SHARE_MOUNT/all_datasets/dcpp_daily/EC-Earth3}
# Output lives under its own dcpp/ subtree rather than beside the ERA5 trees, so
# the two sources are separable at a glance and a stray --aois cannot drop
# hindcast cubes next to reanalysis ones. Note this means era5_features.py's
# DEFAULT path resolution ({root}/{AOI}_processed_data/{source}) will not find
# them: pass --out, as this script does and as the stage-3 examples show.
OUT_ROOT=${OUT_ROOT:-$DATA_ROOT/dcpp}

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RSCRIPT=${RSCRIPT:-Rscript}
PYTHON=${PYTHON:-python}
MODEL=EC-Earth3

AOIS=""; MEMBERS=""; TARGET_YEARS="2002-2022"; INIT_YEARS=""
FORECAST_YEARS="1"; STAGE="both"; GO=0
FEATURES="gdd,edd,tavg,vpd,pr,hot_days"

usage() { sed -n '2,32p' "${BASH_SOURCE[0]}" | sed -E 's/^# ?//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --aois)           AOIS="$2"; shift 2 ;;
    --members)        MEMBERS="$2"; shift 2 ;;
    --target-years)   TARGET_YEARS="$2"; shift 2 ;;
    --init-years)     INIT_YEARS="$2"; shift 2 ;;
    --forecast-years) FORECAST_YEARS="$2"; shift 2 ;;
    --features)       FEATURES="$2"; shift 2 ;;
    --stage)          STAGE="$2"; shift 2 ;;
    --go)             GO=1; shift ;;
    -h|--help)        usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done
[ -n "$AOIS" ] || { echo "--aois is required (e.g. --aois ZM,ET,BF)" >&2; exit 2; }

# ── the share ────────────────────────────────────────────────────────────────
# Mounted is not the same as readable: an SMB share can be present and still
# refuse to list, so it is checked by actually listing it.
if ! mountpoint -q "$SHARE_MOUNT" 2>/dev/null && ! findmnt -rn "$SHARE_MOUNT" >/dev/null 2>&1; then
  echo "$SHARE_MOUNT is not mounted; trying mount ..."
  if ! mount "$SHARE_MOUNT" 2>/dev/null && ! sudo mount "$SHARE_MOUNT"; then
    cat >&2 <<EOF
Could not mount $SHARE_MOUNT.

This script does not carry credentials. Mount it by hand, or give it an
/etc/fstab entry so a bare 'mount $SHARE_MOUNT' works, for example:

  //<synology-host>/CRL_storage  $SHARE_MOUNT  cifs  credentials=/etc/cifs-crl,uid=\$(id -u),gid=\$(id -g),iocharset=utf8,vers=3.0,_netdev  0  0

with the username and password in /etc/cifs-crl (chmod 600).
EOF
    exit 1
  fi
fi
if ! ls "$SHARE_MOUNT" >/dev/null 2>&1; then
  echo "$SHARE_MOUNT is mounted but cannot be listed -- check permissions" >&2; exit 1
fi

for p in "$DATA_ROOT/cy-bench/polygons" \
         "$DATA_ROOT/CroplandPastureArea2000_Geotiff/cropland2000_area.tif" \
         "$DCPP_ROOT"; do
  [ -e "$p" ] || { echo "missing required path: $p" >&2; exit 1; }
done

# ── members ──────────────────────────────────────────────────────────────────
# Discovered from the archive rather than hardcoded, so a newly published
# realisation is picked up without editing this file.
if [ -z "$MEMBERS" ]; then
  MEMBERS=$(find "$DCPP_ROOT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null \
            | grep -E '^r[0-9]+i[0-9]+p[0-9]+f[0-9]+$' | sort -V | paste -sd, -)
  [ -n "$MEMBERS" ] || { echo "no ensemble members found under $DCPP_ROOT" >&2; exit 1; }
fi

# ── the plan ─────────────────────────────────────────────────────────────────
IFS=',' read -ra AOI_LIST  <<< "$AOIS"
IFS=',' read -ra MEM_LIST  <<< "$MEMBERS"
IFS=',' read -ra FY_LIST   <<< "$FORECAST_YEARS"

init_range_for_fy() {   # target lo-hi, lead -> "lo-hi" of initialisation years
  local lo=${TARGET_YEARS%%-*} hi=${TARGET_YEARS##*-} fy=$1
  echo "$((lo - fy))-$((hi - fy))"
}

echo "DCPP -> per-pixel cubes + features"
echo "  share      : $SHARE_MOUNT"
echo "  cy-bench   : $DATA_ROOT/cy-bench"
echo "  archive    : $DCPP_ROOT"
echo "  output     : $OUT_ROOT/{AOI}_processed_data/${MODEL}_{member}"
echo "  AOIs       : ${AOI_LIST[*]}"
echo "  members    : ${MEM_LIST[*]}"
echo "  lead times : ${FY_LIST[*]}"
echo
printf '  %-4s %-12s %s\n' FY "init years" "target years"
total=0
for fy in "${FY_LIST[@]}"; do
  if [ -n "$INIT_YEARS" ]; then ir=$INIT_YEARS
                                tl=$(( ${INIT_YEARS%%-*} + fy )); th=$(( ${INIT_YEARS##*-} + fy ))
  else ir=$(init_range_for_fy "$fy"); tl=${TARGET_YEARS%%-*}; th=${TARGET_YEARS##*-}; fi
  n=$(( ${ir##*-} - ${ir%%-*} + 1 ))
  printf '  %-4s %-12s %s   (%d years)\n' "$fy" "$ir" "$tl-$th" "$n"
  total=$(( total + n ))
done
jobs_n=$(( ${#AOI_LIST[@]} * ${#MEM_LIST[@]} * ${#FY_LIST[@]} ))
echo
echo "  $jobs_n stage-1 job(s), $(( total * ${#AOI_LIST[@]} * ${#MEM_LIST[@]} )) AOI-years in total"

if [ "$GO" -ne 1 ]; then
  echo
  echo "Plan only. Re-run with --go to start. Inside tmux is advisable:"
  echo "  tmux new -s dcpp"
  exit 0
fi

# ── run ──────────────────────────────────────────────────────────────────────
export AGIMPACTS_ROOT="$DATA_ROOT"
failed=(); done_n=0; t0=$SECONDS

for aoi in "${AOI_LIST[@]}"; do
  for mem in "${MEM_LIST[@]}"; do
    tag="${MODEL}_${mem}"
    out="$OUT_ROOT/${aoi}_processed_data/${tag}"
    log="$out/run_${aoi}_${mem}.log"
    mkdir -p "$out"

    for fy in "${FY_LIST[@]}"; do
      if [ -n "$INIT_YEARS" ]; then years=$INIT_YEARS; else years=$(init_range_for_fy "$fy"); fi
      label="$aoi/$mem/FY$fy"
      printf '[%s] %s  s%s -> %s\n' "$(date +%H:%M:%S)" "$label" "$years" "$out"

      if [ "$STAGE" = "1" ] || [ "$STAGE" = "both" ]; then
        if ! "$RSCRIPT" "$REPO/R/dcpp-pixel-store.R" \
              --aoi="$aoi" --model="$MODEL" --member="$mem" --tag="$tag" \
              --forecast-year="$fy" --data-dir="$DCPP_ROOT/$mem/day" \
              --out="$out" "$years" >>"$log" 2>&1; then
          echo "    stage 1 FAILED -- see $log"; failed+=("$label:stage1"); continue
        fi
      fi

      if [ "$STAGE" = "2" ] || [ "$STAGE" = "both" ]; then
        if ! "$PYTHON" "$REPO/python/era5_features.py" \
              --aoi="$aoi" --source="$tag" --forecast-year="$fy" \
              --years="$years" --stage=features --features="$FEATURES" \
              --out="$out" --root="$DATA_ROOT" >>"$log" 2>&1; then
          echo "    stage 2 FAILED -- see $log"; failed+=("$label:stage2"); continue
        fi
      fi
      done_n=$(( done_n + 1 ))
      echo "    done"
    done
  done
done

printf '\nFinished in %dh%02dm. %d job(s) ok, %d failed.\n' \
  $(( (SECONDS-t0)/3600 )) $(( ((SECONDS-t0)%3600)/60 )) "$done_n" "${#failed[@]}"
if [ "${#failed[@]}" -gt 0 ]; then
  printf '  failed: %s\n' "${failed[*]}"
  echo "  Re-run the same command; finished years are skipped."
  exit 1
fi
