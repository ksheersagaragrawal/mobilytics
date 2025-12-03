# mobilytics
ECE 143 grp 16 project analysing mobile device usage patterns using the Gigasheet dataset. Includes data cleaning, EDA, statistical modeling, and visualizations to explore screen time trends, app usage, OS differences, and demographic risk factors. Aimed at supporting digital well-being and behaviour research.

## Repository layout

```
mobilytics/
├── data/                     # Raw and cleaned CSVs (not tracked in git)
├── notebooks/                # Exploratory and presentation notebooks
├── presentation/             # Slide deck assets
├── src/                      # Project Python modules            
├── tests/                    # Reserved for automated tests
├── visuals/                  # Exported charts or dashboards
└── README.md                 # Project overview and instructions
```

## Environment setup

1. Ensure Python 3.10+ is installed (`python3 --version`).
2. (Optional) Create a virtual environment:
	- `python3 -m venv .venv`
	- `source .venv/bin/activate`
3. Install dependencies:
	- `pip install -r requirements.txt` 

## Running the preprocessing pipeline

The preprocessing logic lives in `src/data_preprocessing.py`. It (load → inspect → validate → clean) and writes a sanitized CSV for downstream analysis.

```
python src/data_preprocessing.py \
	 --input data/user_behavior_dataset.csv \
	 --output data/user_behavior_dataset_cleaned.csv
```

- `--input`: path to the raw Gigasheet export.
- `--output`: destination for the cleaned dataset (created if missing).
- Script prints dataset structure, data quality diagnostics, and final shape.

## Using the cleaned data

- Notebooks should import `data/user_behavior_dataset_cleaned.csv` to ensure consistent EDA, modeling, and visualization results.
- Visuals or dashboards in `visuals/` should cite the cleaned dataset version/date to meet grading criteria on impact and novelty.
- Future additions (e.g., feature engineering, modeling) belong in new modules under `src/` or notebooks inside `notebooks/` for clarity and teamwork transparency.
