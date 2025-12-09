# Insurance Risk Analytics – Week 3 Capstone

Final submission repository for the B8 Insurance Week 3 challenge. It packages exploratory analysis, statistical hypothesis testing, predictive modelling, and narrative reporting for an auto insurance claims portfolio (Feb 2014–Aug 2015).

## Repository Highlights
- **Data enrichment layer** (`src/data_utils.py`) standardises claim totals, engineered premiums, severity tiers, and synthetic geography used across all tasks.
- **Task 1–2**: Reproducible EDA via `src/eda.py`, with interim figures in `reports/images/`.
- **Task 3**: Automated statistical tests in `src/task3_stats.py`, exporting JSON/CSV summaries for provincial and demographic risk hypotheses.
- **Task 4**: Modelling pipeline in `src/task4_modeling.py` training severity/premium regressors, high-severity propensity classifiers, SHAP explanations, and risk-based premium simulations.
- **Final Task**: Synthesises insights and recommendations for underwriting stakeholders.

## Data & Versioning
- Primary dataset: `SM/data/insurance.csv` (retrieved via DVC).
- DVC configured with a local remote (`~/.dvc-storage`) for reproducible pulls: `.env/bin/dvc pull`.
- Generated analytics artefacts (images, plots, CSVs) are versioned under `reports/` for auditability.

## Environment Setup
1. Create a Python 3.13 virtual environment and activate it.
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
3. (Optional) Use the project interpreter directly for commands: `.env/bin/python` | OR you can source your environment.

## Running the Pipelines
| Task | Command | Output |
| --- | --- | --- |
| EDA (Tasks 1–2) | `.env/bin/python src/eda.py` | KPI JSON + plots under `reports/` |
| Hypothesis tests (Task 3) | `.env/bin/python src/task3_stats.py` | `reports/task3_hypothesis_tests.{json,csv}` |
| Modelling & pricing (Task 4) | `.env/bin/python src/task4_modeling.py` | `reports/task4_model_metrics.json`, risk premium CSVs, SHAP plot |

## Key Reports & Artefacts
- `reports/model_predictions_sample.csv`: top policies by premium delta for storytelling.
- `reports/risk_premium_segments.csv`: postal/province level pricing recommendations.
- `reports/images/`: all figures embedded in reports (`severity_feature_importance.png`, `dist_totalclaims.png`, etc.).

## Project Structure
- `src/`: reusable code modules and task-specific pipelines.
- `reports/`: JSON metrics, CSVs, and images used in deliverables.
- `Samples-Technical_Content/`: supplementary material excluded from DVC and not part of submission outputs.
- `.github/workflows/python.yml`: CI smoke test that imports dependencies and validates the EDA script.

## Thank You!