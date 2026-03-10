#!/usr/bin/env bash
# ============================================================
# scripts/run_e001_02.sh — uruchamia profil e001-02 w tmux
#
# Użycie:
#   ./scripts/run_e001_02.sh <profil> [DATA_DIR] [override1=v1 override2=v2 ...]
#
# Przykłady:
#   ./scripts/run_e001_02.sh phase_b_r0_baseline_32
#   ./scripts/run_e001_02.sh smoke
#   ./scripts/run_e001_02.sh overnight /data/celeba
#   ./scripts/run_e001_02.sh fast "" steps=5000 batch_size=64
#
# Sesja tmux: e001-02-<profil>-<HH:MM:SS>
# Logi:       runs/<YYYYMMDD_HHMMSS>_e001-02-<profil>/train.log
#
# Po uruchomieniu terminal podłącza się do sesji.
# Wyjście z sesji:  Ctrl+B, D  (tmux detach) — trening działa dalej.
# Ponowne podłączenie: tmux attach -t <nazwa_sesji>
# ============================================================
set -euo pipefail

PROFILE="${1:-}"
DATA_DIR="${2:-}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true
OVERRIDES=("$@")          # reszta argumentów to overrides

if [[ -z "${PROFILE}" ]]; then
    echo "❌  PROFILE is required."
    echo "    Użycie: $0 <profil> [DATA_DIR] [key=val ...]"
    echo ""
    echo "    Dostępne profile:"
    ls "$(dirname "$0")/../e001-02-r3gan-baseline/src/configs/"*.yaml \
        | xargs -I{} basename {} .yaml | sed 's/^/      /'
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
TIME="$(date +%H%M%S)"
SESSION="e001-02-${PROFILE}-${TIME}"
RUN_DIR="${PROJECT_DIR}/runs/${TS}_e001-02-${PROFILE}"

# Sprawdź czy profil istnieje
CONFIG_SRC="${PROJECT_DIR}/e001-02-r3gan-baseline/src/configs/${PROFILE}.yaml"
if [[ ! -f "${CONFIG_SRC}" ]]; then
    echo "❌  Nie znaleziono profilu: ${CONFIG_SRC}"
    echo ""
    echo "    Dostępne profile:"
    ls "${PROJECT_DIR}/e001-02-r3gan-baseline/src/configs/"*.yaml \
        | xargs -I{} basename {} .yaml | sed 's/^/      /'
    exit 1
fi

# Sprawdź czy sesja o tej samej nazwie już działa
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "⚠️   Sesja '${SESSION}' już istnieje."
    echo "    Podłączam się do istniejącej sesji..."
    tmux attach -t "${SESSION}"
    exit 0
fi

mkdir -p "${RUN_DIR}"
cp -f "${CONFIG_SRC}" "${RUN_DIR}/"

# Buduj komendę run.py
RUN_CMD="python run.py --profile ${PROFILE}"
[[ -n "${DATA_DIR}" ]] && RUN_CMD+=" --data-dir ${DATA_DIR}"
if [[ ${#OVERRIDES[@]} -gt 0 ]]; then
    RUN_CMD+=" --override ${OVERRIDES[*]}"
fi

# Skrypt startowy wewnątrz tmux
START_SH="${RUN_DIR}/start.sh"
cat > "${START_SH}" << STARTEOF
#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR}"
. .venv/bin/activate
. scripts/env_paths.sh

echo "================================================================"
echo "  e001-02-r3gan-baseline"
echo "  Profil  : ${PROFILE}"
echo "  Sesja   : ${SESSION}"
echo "  Run dir : ${RUN_DIR}"
echo "  Czas    : \$(date '+%Y-%m-%d %H:%M:%S')"
[[ -n "${DATA_DIR}" ]] && echo "  DATA_DIR: ${DATA_DIR}"
[[ "${OVERRIDES[*]:-}" != "" ]] && echo "  Overrides: ${OVERRIDES[*]:-}"
echo "================================================================"
echo ""
echo "💡 Wyjście bez zatrzymywania treningu: Ctrl+B, D"
echo ""

cd "${PROJECT_DIR}/e001-02-r3gan-baseline"

${RUN_CMD} 2>&1 | tee "${RUN_DIR}/train.log"

echo ""
echo "✅ Trening zakończony: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "   Logi: ${RUN_DIR}/train.log"
STARTEOF
chmod +x "${START_SH}"

# Uruchom sesję tmux w tle
tmux new-session -d -s "${SESSION}" "${START_SH}"

echo ""
echo "✅  Sesja tmux uruchomiona: ${SESSION}"
echo "📄  Logi:      ${RUN_DIR}/train.log"
echo "📡  Tail:      tail -f ${RUN_DIR}/train.log"
echo "📎  Detach:    Ctrl+B, D"
echo "🔗  Podłącz:   tmux attach -t ${SESSION}"
echo ""

# Podłącz terminal do sesji (wyjście = Ctrl+B D, trening działa dalej)
tmux attach -t "${SESSION}"

