# End-to-End Insurance Risk Analytics (Week 3)

This repository contains the interim deliverables for AlphaCare Insurance Solutions (ACIS) Week 3 challenge: EDA and DVC setup on historical car insurance claims data (Feb 2014–Aug 2015).

## Objectives
- Task 1: Git, CI, and foundational EDA focusing on loss ratio and risk patterns.
- Task 2: Reproducible data pipeline using Data Version Control (DVC) with local remote.

## Project Structure
- `SM/data/insurance.csv`: Working dataset for EDA and DVC tracking.
- `src/eda.py`: Reproducible EDA script generating KPIs and plots.
- `notebooks/eda_interim.ipynb`: Notebook for exploration and screenshots.
- `reports/interim_report.md`: Interim write-up covering Task 1–2.
- `.github/workflows/python.yml`: Minimal CI to validate environment and run EDA.

## Quick Start
1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Run EDA script or open the notebook.

## DVC
- Initialized with a local remote for reproducible data tracking.
