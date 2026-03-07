#!/usr/bin/env bash
set -euo pipefail

# Użycie:
#   ./scripts/run_tmux.sh exp1 configs/exp1.yaml "python train.py --config configs/exp1.yaml"
#
# Nazwa sesji tmux: <project>-<exp>-<timestamp>
# Logi: runs/<timestamp>_<exp>/train.log

EXP_NAME="${1:-exp}"
CONFIG_PATH="${2:-}"
CMD="${3:-}"

if [[ -z "${CONFIG_PATH}" || -z "${CMD}" ]]; then
  echo "Usage: $0 <exp_name> <config_path> <command>"
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_NAME="$(basename "${PROJECT_DIR}")"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${PROJECT_DIR}/runs/${TS}_${EXP_NAME}"
SESSION="${PROJECT_NAME}-${EXP_NAME}-${TS}"

mkdir -p "${RUN_DIR}"

# Skrypt startowy uruchamiany wewnątrz tmux
START_SH="${RUN_DIR}/start.sh"
cat > "${START_SH}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_DIR}"
source scripts/env_paths.sh
source .venv/bin/activate

echo "=== RUN ${SESSION} ==="
echo "PWD: \$(pwd)"
echo "Python: \$(which python)"
echo "Config: ${CONFIG_PATH}"
echo "Cmd: ${CMD}"
echo "Time: \$(date -Is)"
echo

# kopiuj config do run_dir dla powtarzalności
cp -f "${CONFIG_PATH}" "${RUN_DIR}/"

# właściwy trening (log do pliku)
${CMD} 2>&1 | tee "${RUN_DIR}/train.log"
EOF

chmod +x "${START_SH}"

tmux new-session -d -s "${SESSION}" "${START_SH}"
echo "Started tmux session: ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Logs:   ${RUN_DIR}/train.log"