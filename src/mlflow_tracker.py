"""
src/mlflow_tracker.py — Centralized MLflow experiment tracking.

Design goals:
- Easy to use: one import, automatic setup
- Resilient: works offline, survives crashes, can resume
- Complete: logs params, metrics, artifacts, tags
- Flexible: works with both simple split and nested CV modes

Usage:
    from src.mlflow_tracker import tracker

    # In run_all.py — start experiment
    tracker.start_experiment("SAE_Prediction", nested=True, use_text=True, tune=True)

    # Per model-phase — start a run
    with tracker.start_run("XGBoost", phase="1"):
        tracker.log_params({"n_estimators": 100, "max_depth": 6})
        # ... train model ...
        tracker.log_metrics({"roc_auc": 0.88, "f1": 0.77})
        tracker.log_artifact("results/XGBoost_1.json")

    # At the end
    tracker.log_aggregated_results("results/aggregated_results.json")

View results:
    mlflow ui --backend-store-uri file:///path/to/mlruns

Or use the helper:
    tracker.launch_ui()
"""

import os
import sys
import json
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# ── MLflow setup (optional — works offline if not installed) ──────
# Use local SQLite backend (MLflow 3.x requires database backend)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB_PATH = _PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB_PATH}"
)
# Allow artifacts to be stored locally
MLFLOW_ARTIFACT_ROOT = os.environ.get(
    "MLFLOW_ARTIFACT_ROOT",
    str(_PROJECT_ROOT / "mlartifacts")
)

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.environ.setdefault("MLFLOW_ARTIFACT_ROOT", MLFLOW_ARTIFACT_ROOT)
    _MLFLOW_AVAILABLE = True
    # Suppress verbose MLflow warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
except ImportError:
    mlflow = None
    _MLFLOW_AVAILABLE = False


class MLFlowTracker:
    """
    Centralized experiment tracker for SAE prediction project.

    Handles the full lifecycle:
    - Experiment creation / reuse
    - Nested runs (one per model-phase combination)
    - Per-fold metrics (in nested CV)
    - Training curves (for DL models)
    - Hyperparameter tuning results
    - Artifact logging (JSON results, checkpoints)
    - Aggregation logging
    """

    def __init__(self):
        self._experiment_name = None
        self._experiment_id = None
        self._active_run = None
        self._run_stack = []  # For nested runs
        self._enabled = _MLFLOW_AVAILABLE
        if not _MLFLOW_AVAILABLE:
            print("  [MLflow] Not installed — tracking disabled. Install with: pip install mlflow")

    # ── Enable / Disable ────────────────────────────────────────────

    def disable(self):
        """Disable tracking (no-op all operations)."""
        self._enabled = False

    def enable(self):
        """Re-enable tracking."""
        self._enabled = True

    @property
    def enabled(self):
        return self._enabled

    # ── Experiment Management ───────────────────────────────────────

    def start_experiment(self, name="SAE_Prediction", **tags):
        """
        Start or resume an experiment.

        Parameters
        ----------
        name : str — experiment name (default: "SAE_Prediction")
        **tags : extra tags to set on the experiment (e.g. nested=True)
        """
        if not self._enabled or mlflow is None:
            return

        self._experiment_name = name

        # Create experiment if it doesn't exist, otherwise reuse
        try:
            self._experiment_id = mlflow.create_experiment(name)
        except Exception:
            exp = mlflow.get_experiment_by_name(name)
            self._experiment_id = exp.experiment_id if exp else None

        if self._experiment_id is None:
            print("  [MLflow] Warning: could not create or find experiment")
            return

        mlflow.set_experiment(name)

        # Set experiment-level tags
        with mlflow.start_run(run_name="__experiment_meta__", experiment_id=self._experiment_id):
            mlflow.set_tag("start_time", datetime.now().isoformat())
            mlflow.set_tag("python_version", sys.version.split()[0])
            for k, v in tags.items():
                mlflow.set_tag(k, str(v))

        print(f"  [MLflow] Experiment: '{name}' (ID: {self._experiment_id})")
        print(f"  [MLflow] Tracking URI: {MLFLOW_TRACKING_URI}")

    # ── Run Management ──────────────────────────────────────────────

    @contextmanager
    def start_run(self, model_name, phase=None, run_name=None, nested=False):
        """
        Start a tracked run for a model (optionally a specific phase).

        Usage:
            with tracker.start_run("XGBoost", phase="1"):
                # training code
                tracker.log_metrics({"roc_auc": 0.88})
        """
        if not self._enabled:
            yield None
            return

        run_name = run_name or f"{model_name}_phase{phase}" if phase else model_name

        run = mlflow.start_run(
            run_name=run_name,
            experiment_id=self._experiment_id,
            nested=nested,
        )
        self._active_run = run
        self._run_stack.append(run)

        # Set tags
        mlflow.set_tag("model_name", model_name)
        if phase:
            mlflow.set_tag("phase", str(phase))
        mlflow.set_tag("mode", "nested_cv" if nested else "simple_split")

        try:
            yield run
        finally:
            mlflow.end_run()
            self._run_stack.pop()
            self._active_run = self._run_stack[-1] if self._run_stack else None

    @contextmanager
    def start_fold_run(self, model_name, phase, fold_idx, total_folds, parent_run_id=None):
        """
        Start a child run for a single fold in nested CV.
        Logs under the parent model-phase run.

        Usage:
            with tracker.start_run("XGBoost", phase="1"):
                for fold in range(5):
                    with tracker.start_fold_run("XGBoost", "1", fold, 5):
                        # fold training
                        tracker.log_metrics({"roc_auc": 0.88})
        """
        if not self._enabled:
            yield None
            return

        run_name = f"fold_{fold_idx}"

        run = mlflow.start_run(
            run_name=run_name,
            experiment_id=self._experiment_id,
            nested=True,
        )
        self._run_stack.append(run)
        self._active_run = run

        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("phase", str(phase))
        mlflow.set_tag("fold", fold_idx)
        mlflow.set_tag("total_folds", total_folds)
        mlflow.set_tag("run_type", "fold")
        if parent_run_id:
            mlflow.set_tag("parent_run_id", parent_run_id)

        try:
            yield run
        finally:
            mlflow.end_run()
            self._run_stack.pop()
            self._active_run = self._run_stack[-1] if self._run_stack else None

    # ── Logging: Parameters ─────────────────────────────────────────

    def log_params(self, params: dict):
        """Log a dictionary of parameters."""
        if not self._enabled or not params:
            return
        # Flatten nested dicts and convert to strings
        flat = self._flatten_dict(params)
        for k, v in flat.items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                pass  # Silently skip params that can't be logged

    def log_param(self, key, value):
        """Log a single parameter."""
        if not self._enabled:
            return
        try:
            mlflow.log_param(key, value)
        except Exception:
            pass

    # ── Logging: Metrics ────────────────────────────────────────────

    def log_metrics(self, metrics: dict, step=None):
        """Log a dictionary of metrics (optionally at a step)."""
        if not self._enabled or not metrics:
            return
        for k, v in metrics.items():
            self.log_metric(k, v, step=step)

    def log_metric(self, key, value, step=None):
        """Log a single metric value."""
        if not self._enabled or value is None:
            return
        try:
            # Only log numeric values
            if isinstance(value, (int, float)) and not (isinstance(value, float) and (value != value)):
                mlflow.log_metric(key, value, step=step)
        except Exception:
            pass

    # ── Logging: Training Curves ────────────────────────────────────

    def log_training_curve(self, history: list, prefix=""):
        """
        Log training history as step-wise metrics.

        Parameters
        ----------
        history : list of dicts — e.g. [{"train_loss": 0.5, "val_loss": 0.4}, ...]
        prefix  : str — prefix for metric names (e.g. "inner_" for inner CV)
        """
        if not self._enabled or not history:
            return
        for epoch, record in enumerate(history):
            for key, value in record.items():
                metric_name = f"{prefix}{key}" if prefix else key
                self.log_metric(metric_name, value, step=epoch)

    # ── Logging: Artifacts ──────────────────────────────────────────

    def log_artifact(self, path: str):
        """Log a file as an artifact."""
        if not self._enabled or not os.path.exists(path):
            return
        try:
            mlflow.log_artifact(path)
        except Exception as e:
            print(f"  [MLflow] Warning: could not log artifact {path}: {e}")

    def log_artifacts(self, directory: str):
        """Log all files in a directory as artifacts."""
        if not self._enabled or not os.path.isdir(directory):
            return
        try:
            mlflow.log_artifacts(directory)
        except Exception as e:
            print(f"  [MLflow] Warning: could not log artifacts from {directory}: {e}")

    def log_json_artifact(self, data: dict, filename: str):
        """Log a dictionary as a JSON artifact file."""
        if not self._enabled:
            return
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2)
            temp_path = f.name
        try:
            mlflow.log_artifact(temp_path, artifact_path="json_artifacts")
        finally:
            os.unlink(temp_path)

    # ── Logging: Model Info ─────────────────────────────────────────

    def log_model_info(self, model_name: str, params: dict, is_dl: bool = False):
        """Log model architecture/hyperparameters as params."""
        if not self._enabled:
            return
        mlflow.set_tag("model_type", "dl" if is_dl else "sklearn")
        self.log_params({f"model.{k}": v for k, v in params.items()})

    def log_tuning_results(self, best_params: dict, tuning_method: str = "optuna"):
        """Log hyperparameter tuning results."""
        if not self._enabled or not best_params:
            return
        mlflow.set_tag("tuning_method", tuning_method)
        mlflow.set_tag("tuning_enabled", "true")
        self.log_params({f"tuned.{k}": v for k, v in best_params.items()})

    # ── Logging: Config ─────────────────────────────────────────────

    def log_config(self):
        """Log all config constants as parameters."""
        if not self._enabled:
            return
        from src import config
        cfg = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}
        self.log_params({f"config.{k}": v for k, v in cfg.items()})

    # ── Logging: Aggregated Results ─────────────────────────────────

    def log_aggregated_results(self, aggregated_data: list):
        """
        Log aggregated results (mean ± std per model-phase).

        Parameters
        ----------
        aggregated_data : list of dicts — from aggregate_results.py
        """
        if not self._enabled or not aggregated_data:
            return

        # Log as JSON artifact
        self.log_json_artifact(aggregated_data, "aggregated_results.json")

        # Log best model per phase as tags
        for row in aggregated_data:
            phase = row.get("phase")
            model = row.get("model")
            roc_auc = row.get("roc_auc_mean")
            if phase and model and roc_auc is not None:
                mlflow.set_tag(f"best_phase_{phase}", model)
                mlflow.log_metric(f"best_roc_auc_phase_{phase}", roc_auc)

    # ── Query / Resume Helpers ──────────────────────────────────────

    def get_run_id(self):
        """Get the current active run ID."""
        if self._active_run:
            return self._active_run.info.run_id
        return None

    def get_completed_folds(self, model_name: str, phase: str) -> set:
        """
        Get the set of completed fold indices for a model-phase.
        Useful for resuming nested CV after a crash.
        """
        if not self._enabled or not self._experiment_id:
            return set()

        try:
            runs = mlflow.search_runs(
                experiment_ids=[self._experiment_id],
                filter_string=(
                    f"tags.model_name = '{model_name}' and "
                    f"tags.phase = '{phase}' and "
                    f"tags.run_type = 'fold'"
                ),
            )
            if runs.empty:
                return set()
            return set(runs["tags.fold"].astype(int))
        except Exception:
            return set()

    # ── UI Launch ───────────────────────────────────────────────────

    def launch_ui(self, port=5000):
        """Launch the MLflow UI in a background process."""
        if mlflow is None:
            print("  [MLflow] Not installed — cannot launch UI. Install with: pip install mlflow")
            return
        import subprocess
        uri = MLFLOW_TRACKING_URI.replace("file://", "")
        cmd = ["mlflow", "ui", "--backend-store-uri", uri, "--port", str(port)]
        print(f"  [MLflow] Starting UI on http://localhost:{port}")
        print(f"  [MLflow] Command: {' '.join(cmd)}")
        subprocess.Popen(cmd)

    # ── Internal Helpers ────────────────────────────────────────────

    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLFlowTracker._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


# ── Singleton instance ────────────────────────────────────────────
tracker = MLFlowTracker()
