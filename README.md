# mobilytics
ECE 143 grp 16 project analysing mobile device usage patterns using the Gigasheet dataset. Includes data cleaning, EDA, statistical modeling, and visualizations to explore screen time trends, app usage, OS differences, and demographic risk factors.

## Repository layout

```
mobilytics/
├── data/                     # Raw and cleaned CSVs; required plots saved here
├── notebooks/                # EDA_visualizations.ipynb (+ archive of older notebooks)
├── presentation/             # final_presentation.pdf
├── src/                      # Project Python package (preprocessing, clustering, regression, viz)
├── requirements.txt          # Third-party dependencies
└── README.md                 # Project overview and instructions
```

## Environment setup

1. Ensure Python 3.10+ is installed (`python3 --version`).
2. (Optional) create a virtual environment:
   - `python3 -m venv .venv`
   - `source .venv/bin/activate`
3. Install dependencies:
   - `pip install -r requirements.txt`

## Running the code

- **Preprocess the dataset (clean CSV):**
  ```
  python -m src.data_preprocessing \
    --input data/user_behavior_dataset.csv \
    --output data/user_behavior_dataset_cleaned.csv
  ```

- **Clustering pipeline (saves elbow + cluster plots to data/):**
  ```
  python -m src.clustering
  ```

- **Regression pipeline (Random Forest + XGBoost; saves feature plots to data/):**
  ```
  python -m src.regression
  ```

- **Notebook:** open `notebooks/EDA_visualizations.ipynb` to view all visuals used in the presentation (older notebooks are kept in `notebooks/archive/` for reference).

## Third-party modules

Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, SHAP (full versions pinned in `requirements.txt`).
