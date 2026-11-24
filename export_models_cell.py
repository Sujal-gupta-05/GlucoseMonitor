# RUN THIS CELL AT THE END OF hypo_statistical_reliability_corrected.ipynb
# It fits and saves the exact feature-compatible models needed by the demo app.
# It uses fit_df for validation-based regression selection, then refits on train_df.
# Classification thresholds are taken from validation-only threshold selection.

from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error

required_vars = ["FULL", "fit_df", "train_df", "val_df", "CLF_CFGS", "CLF_CHOICE", "THR"]
missing_vars = [name for name in required_vars if name not in globals()]
if missing_vars:
    raise RuntimeError(
        f"Missing notebook variables: {missing_vars}. "
        "Run the corrected notebook from top to bottom first."
    )

SEED = 42
REG_CONFIGS = [
    dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15),
    dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50),
    dict(max_iter=500, learning_rate=0.03, max_leaf_nodes=15, l2_regularization=1.0),
    dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, loss="absolute_error"),
]

bundle = {
    "features": list(FULL),
    "models": {},
    "thresholds": {},
    "metadata": {
        "seed": SEED,
        "hypoglycemia_threshold_mg_dl": 70,
        "sampling_minutes": 5,
        "training_note": "Regression config selected by validation RMSE; classifier config selected by validation AP; final models refit on train_df.",
    },
}

for horizon in (30, 60):
    target = f"glucose_{horizon}m"

    # Tune regression on fit_df -> val_df; refit selected configuration on train_df.
    best_cfg, best_rmse = None, float("inf")
    for cfg in REG_CONFIGS:
        candidate = HistGradientBoostingRegressor(random_state=SEED, **cfg)
        y_fit = fit_df[target].to_numpy() - fit_df["glucose"].to_numpy()
        candidate.fit(fit_df[FULL], y_fit)
        pred_val = candidate.predict(val_df[FULL]) + val_df["glucose"].to_numpy()
        rmse = mean_squared_error(val_df[target], pred_val) ** 0.5
        if rmse < best_rmse:
            best_cfg, best_rmse = cfg, rmse

    reg_model = HistGradientBoostingRegressor(random_state=SEED, **best_cfg)
    y_train = train_df[target].to_numpy() - train_df["glucose"].to_numpy()
    reg_model.fit(train_df[FULL], y_train)
    bundle["models"][f"regression_{horizon}m"] = reg_model
    bundle["metadata"][f"regression_{horizon}m_validation_rmse"] = float(best_rmse)
    bundle["metadata"][f"regression_{horizon}m_config"] = best_cfg

    # Classifier: selected configuration and threshold from the executed notebook.
    choice = CLF_CHOICE[(horizon, "HGB classifier")]
    clf_cfg = CLF_CFGS[choice]
    clf_train = train_df[train_df["elig"]].copy()
    clf_model = HistGradientBoostingClassifier(random_state=SEED, **clf_cfg)
    clf_model.fit(clf_train[FULL], clf_train[f"hypo_{horizon}m"].astype(int))
    bundle["models"][f"classifier_{horizon}m"] = clf_model

    threshold = float(THR[(horizon, "HGB classifier", "recall >= 80%")])
    bundle["thresholds"][f"{horizon}m"] = threshold
    bundle["metadata"][f"classifier_{horizon}m_config"] = clf_cfg
    bundle["metadata"][f"classifier_{horizon}m_validation_config_index"] = int(choice)

artifact_path = Path.cwd() / "glucosemonitor_models.joblib"
joblib.dump(bundle, artifact_path, compress=3)

print("Saved demo model bundle:", artifact_path.resolve())
print("Feature count:", len(bundle["features"]))
for h in (30, 60):
    print(
        f"{h}m | regression validation RMSE: "
        f"{bundle['metadata'][f'regression_{h}m_validation_rmse']:.2f} mg/dL"
        f" | alert threshold: {bundle['thresholds'][f'{h}m']:.4f}"
    )
print("Model export complete. Copy glucosemonitor_models.joblib into the folder containing app.py.")
