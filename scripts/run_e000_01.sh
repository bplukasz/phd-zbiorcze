#!/usr/bin/env bash
# ============================================================
# scripts/run_e000_01.sh - uruchamia profil e000-01 w tmux
#
# Uzycie:
#   ./scripts/run_e000_01.sh <profil> [DATA_DIR] [override1=v1 override2=v2 ...]
#
# Przyklady:
#   ./scripts/run_e000_01.sh fast
#   ./scripts/run_e000_01.sh smoke
#   ./scripts/run_e000_01.sh overnight /data/celeba
#   ./scripts/run_e000_01.sh fast "" steps=5000 batch_size=64
#
# Sesja tmux: e000-01-<profil>-<HHMMSS>
# Logi:       runs/<YYYYMMDD_HHMMSS>_e000-01-<profil>/train.log
#
# Po uruchomieniu terminal podlacza sie do sesji.
# Wyjscie z sesji:  Ctrl+B, D  (tmux detach) - trening dziala dalej.
# Ponowne podlaczenie: tmux attach -t <nazwa_sesji>
# ============================================================
set -euo pipefail

PROFILE="${1:-}"
DATA_DIR="${2:-}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true
OVERRIDES=("$@")

if [[ -z "${PROFILE}" ]]; then
    echo "ERROR: PROFILE is required."
    echo "Usage: $0 <profil> [DATA_DIR] [key=val ...]"
    echo ""
    echo "Available profiles:"
    ls "$(dirname "$0")/../e000-01-r3gan-baseline/src/configs/"*.yaml \
        | xargs -I{} basename {} .yaml | sed 's/^/  /'
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
TIME="$(date +%H%M%S)"
SESSION="e000-01-${PROFILE}-${TIME}"
RUN_DIR="${PROJECT_DIR}/runs/${TS}_e000-01-${PROFILE}"

CONFIG_SRC="${PROJECT_DIR}/e000-01-r3gan-baseline/src/configs/${PROFILE}.yaml"
if [[ ! -f "${CONFIG_SRC}" ]]; then
    echo "ERROR: Profile not found: ${CONFIG_SRC}"
    echo ""
    echo "Available profiles:"
    ls "${PROJECT_DIR}/e000-01-r3gan-baseline/src/configs/"*.yaml \
        | xargs -I{} basename {} .yaml | sed 's/^/  /'
    exit 1
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "WARN: Session '${SESSION}' already exists."
    echo "Attaching to existing session..."
    tmux attach -t "${SESSION}"
    exit 0
fi

mkdir -p "${RUN_DIR}"
cp -f "${CONFIG_SRC}" "${RUN_DIR}/"

RUN_CMD="python run.py --profile ${PROFILE}"
[[ -n "${DATA_DIR}" ]] && RUN_CMD+=" --data-dir ${DATA_DIR}"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
    RUN_CMD+=" --override ${OVERRIDES[*]}"
fi

START_SH="${RUN_DIR}/start.sh"
cat > "${START_SH}" << STARTEOF
#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR}"
. .venv/bin/activate
. scripts/env_paths.sh

echo "================================================================"
echo "  e000-01-r3gan-baseline"
echo "  Profil  : ${PROFILE}"
echo "  Sesja   : ${SESSION}"
echo "  Run dir : ${RUN_DIR}"
echo "  Czas    : \$(date '+%Y-%m-%d %H:%M:%S')"
[[ -n "${DATA_DIR}" ]] && echo "  DATA_DIR: ${DATA_DIR}"
[[ "${OVERRIDES[*]:-}" != "" ]] && echo "  Overrides: ${OVERRIDES[*]:-}"
echo "================================================================"
echo ""
echo "Tip: detach without stopping training: Ctrl+B, D"
echo ""

cd "${PROJECT_DIR}/e000-01-r3gan-baseline"

${RUN_CMD} 2>&1 | tee "${RUN_DIR}/train.log"

echo ""
echo "Training finished: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "Log: ${RUN_DIR}/train.log"
STARTEOF
chmod +x "${START_SH}"

tmux new-session -d -s "${SESSION}" "${START_SH}"

echo ""
echo "Session started: ${SESSION}"
echo "Log:   ${RUN_DIR}/train.log"
echo "Tail:  tail -f ${RUN_DIR}/train.log"
echo "Detach: Ctrl+B, D"
echo "Attach: tmux attach -t ${SESSION}"
echo ""

tmux attach -t "${SESSION}"

