# GlucoseMonitor — college demonstration app

This is a local Streamlit dashboard for retrospective demonstration of the project's
30/60-minute glucose forecasts and hypoglycemia-risk scores.

## 1. Folder layout

Keep the project and dataset as sibling folders:

```text
Desktop/
├── ML_Project/
│   ├── app.py
│   ├── requirements.txt
│   ├── export_models_cell.py
│   └── glucosemonitor_models.joblib   # created by the notebook export cell
└── HUPA-UCM Diabetes Dataset/
    └── Preprocessed/
        ├── HUPA0001P.csv
        └── ... (25 participant CSV files)
```

## 2. Export the trained models (one-time)

1. Open `hypo_statistical_reliability_corrected.ipynb`.
2. Run all cells from top to bottom.
3. Open `export_models_cell.py`, copy its contents into a new notebook cell at the end,
   and run it.
4. It creates `glucosemonitor_models.joblib` in the notebook's current working directory.
5. Copy that file into the same folder as `app.py`.

The export cell selects regression configurations on validation RMSE, selects the
classifier configurations and alert thresholds using the notebook's validation-only
selection, then fits the final models on the training split.

## 3. Install and launch

From Terminal:

```bash
cd ~/Desktop/ML_Project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit opens a local browser tab. If it does not, open the localhost URL printed
in Terminal (usually http://localhost:8501).

## 4. Demonstration

- Select a participant.
- Move the reading slider to a historical timestamp.
- Show the 30- and 60-minute predicted glucose values.
- Show the model's hypoglycemia-risk probabilities and validation-selected thresholds.
- Expand "Retrospective ground truth" to compare the prediction with subsequent recorded
  glucose values. Explain that future readings are shown only for retrospective comparison,
  never used as model inputs.

## Important limitations

- This is a research prototype, not a medical device or clinical decision-support tool.
- It is not clinically validated and must not be used for treatment decisions.
- The dataset contains 25 people with Type 1 diabetes; generalization is unknown.
- This dashboard is a retrospective demonstration using historical participant data.
