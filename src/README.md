### Environment setup
- `python3 -m venv venv`
- `source venv/bin/activate`
- `pip install -r ../requirements.txt`

### Running modules from repo root
- Preprocess data: `python -m src.data_preprocessing --input data/user_behavior_dataset.csv --output data/user_behavior_dataset_cleaned.csv`
- Clustering visuals: `python -m src.clustering`
- Regression visuals: `python -m src.regression`

Deactivate when done: `deactivate`
