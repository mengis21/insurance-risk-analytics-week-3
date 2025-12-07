# Interim Report — Week 3 (Tasks 1–2)

## Scope
- Task 1: Git setup, foundational EDA, CI.
- Task 2: DVC initialization, local remote, data tracking.

## Data & Setup
- Dataset: `SM/data/insurance.csv` (Feb 2014–Aug 2015 structure).
- Environment: Python 3.11; dependencies in `requirements.txt`.
- CI: GitHub Actions validates env and runs a smoke EDA.
- DVC: Initialized with local remote for auditable data versions.

## EDA Highlights (Interim)
- Overall Loss Ratio (TotalClaims/TotalPremium): [auto-computed, see `reports/kpis_interim.json`].
- Segment insights:
  - Loss Ratio by Province, VehicleType, Gender.
  - Distributions of TotalPremium and TotalClaims with outlier detection via histograms.
  - Monthly trend: TotalPremium vs TotalClaims lines.
  - Correlation matrix for key financial variables.

## Screenshots (attach in submission)
- `reports/images/loss_ratio_by_province.png`
- `reports/images/loss_ratio_by_vehicletype.png`
- `reports/images/loss_ratio_by_gender.png`
- `reports/images/dist_totalpremium.png`
- `reports/images/dist_totalclaims.png`
- `reports/images/monthly_premium_claims.png`
- `reports/images/correlation_matrix.png`
- Repo CI page: successful workflow run on `main`.
- GitHub PR page: `task-1` and `task-2` merges into `main`.

## DVC Setup Summary
- `dvc init` completed.
- Local remote `localstorage` configured.
- `dvc add SM/data/insurance.csv` tracked.
- `dvc push` uploaded to local remote.

## Next Steps
- Extend EDA to statistical testing (Task 3).
- Prepare hypothesis test design and segmentation.

## References
- Cornerstone Insurance Glossary (50 Common Insurance Terms).
- Standard EDA and statistical techniques (histograms, box plots, correlation).
