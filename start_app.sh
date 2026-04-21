# Note: MLflow 3.x - model is served via direct artifact path (stages API removed).
# Press Ctrl+C to stop all three services cleanly.

set -euo pipefail

# Resolve to the directory that contains this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Working directory : $SCRIPT_DIR"

# Activate virtual environment
VENV_DIR=e:/LocalPortfolio/Education/INFNET/aulasDataScience/ds_env
if [[ -f "$VENV_DIR/Scripts/activate" ]]; then
    source "$VENV_DIR/Scripts/activate"
    echo "Virtual env       : $VENV_DIR"
elif [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
    echo "Virtual env       : $VENV_DIR"
else
    echo "WARNING: venv not found at $VENV_DIR -- assuming mlflow/streamlit are on PATH."
fi

# Configuration
TRACKING_PORT=5000
MODEL_PORT=5001
STREAMLIT_PORT=8501
MODEL_NAME="california-housing-best"
MLRUNS_DIR="$SCRIPT_DIR/mlruns"
# Windows-style path for the file:// URI (MLflow rejects bare E:/... as unknown URI scheme)
WIN_MLRUNS_DIR="$(cd "$MLRUNS_DIR" && pwd -W)"
LOG_DIR="$SCRIPT_DIR/.logs"
mkdir -p "$LOG_DIR"
