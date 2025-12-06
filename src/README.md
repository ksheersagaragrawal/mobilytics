### Environment setup
- `python3 -m venv venv`
- `source venv/bin/activate`
- `pip install -r ../requirements.txt`

### Running modules from repo root
- Preprocess data: `python data_preprocessing.py --input data/user_behavior_dataset.csv --output data/user_behavior_dataset_cleaned.csv`
- Clustering visuals: `python clustering.py`
- Regression visuals: `python regression.py`

Deactivate when done: `deactivate`
