"""Utilities to extract, validate, and clean the User Behavior dataset.

This module mirrors the exploratory data-prep workflow that originally lived in a
Google Colab notebook. It loads the raw CSV, performs the documented integrity
checks (duplicates, categorical typos, numeric bounds), removes problematic
records, and exports a tidy version ready for downstream analysis and
visualization work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn. preprocessing import StandardScaler

# Display full columns for quicker inspection when running as a script.
pd.set_option("display.max_columns", None)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "user_behavior_dataset.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "user_behavior_dataset_cleaned.csv"

NUMERIC_COLUMNS = [
	"App Usage Time (min/day)",
	"Screen On Time (hours/day)",
	"Battery Drain (mAh/day)",
	"Number of Apps Installed",
	"Data Usage (MB/day)",
	"Age",
	"User Behavior Class",
]

RANDOM_STATE = 42
FEATURE_COLS = [
	"App Usage Time (min/day)",
	"Screen On Time (hours/day)",
	"Battery Drain (mAh/day)",
	"Number of Apps Installed",
	"Data Usage (MB/day)",
]


def load_dataset(file_path: Path) -> pd.DataFrame:
	"""Return the dataset as a DataFrame and raise a helpful error if missing."""

	if not file_path.exists():
		raise FileNotFoundError(
			f"Dataset not found at {file_path}. Ensure the CSV is present or pass a custom path."
		)
	return pd.read_csv(file_path)


def inspect_dataset(device_usage: pd.DataFrame) -> None:
	"""Print structural metadata to mirror the notebook walkthrough."""

	print("First 6 rows:")
	print(device_usage.head(6))
	print("\nColumn names:")
	print(device_usage.columns.tolist())
	print("\nData types:")
	print(device_usage.dtypes)
	print("\nDataset shape:")
	print(device_usage.shape)


def _print_value_counts(device_usage: pd.DataFrame, columns: Iterable[str]) -> None:
	for col in columns:
		print(f"\n{col} counts:")
		print(device_usage[col].value_counts())


def check_data_validity(device_usage: pd.DataFrame) -> None:
	"""Report duplicate IDs, categorical distributions, and numeric bounds."""

	duplicate_count = device_usage["User ID"].duplicated().sum()
	print(f"Number of duplicate User IDs: {duplicate_count}")

	_print_value_counts(
		device_usage,
		["Device Model", "Operating System", "Gender"],
	)

	for col in NUMERIC_COLUMNS:
		invalid_negative = device_usage[device_usage[col] < 0]
		print(f"Negative values in '{col}': {len(invalid_negative)}")

	behavior_mask = ~device_usage["User Behavior Class"].between(1, 5, inclusive="both")
	print(f"Invalid User Behavior Class values: {behavior_mask.sum()}")

	invalid_app_usage = device_usage["App Usage Time (min/day)"] > 24 * 60
	print(f"Invalid App Usage Time (> 1440 min/day): {invalid_app_usage.sum()}")

	invalid_screen_time = device_usage["Screen On Time (hours/day)"] > 24
	print(f"Invalid Screen On Time (> 24 hours): {invalid_screen_time.sum()}")


def clean_dataset(device_usage: pd.DataFrame) -> pd.DataFrame:
	"""Drop nulls and enforce numeric constraints required for analysis."""

	print("\nMissing values before cleaning:")
	print(device_usage.isna().sum())

	cleaned = device_usage.dropna().copy()

	# Enforce domain rules by filtering out-of-range values.
	cleaned = cleaned[cleaned[NUMERIC_COLUMNS].ge(0).all(axis=1)]
	cleaned = cleaned[cleaned["App Usage Time (min/day)"] <= 24 * 60]
	cleaned = cleaned[cleaned["Screen On Time (hours/day)"] <= 24]
	cleaned = cleaned[cleaned["User Behavior Class"].between(1, 5, inclusive="both")]

	# Ages outside [10, 90] are likely data-entry mistakes for this study.
	cleaned = cleaned[cleaned["Age"].between(10, 90, inclusive="both")]

	print("\nMissing values after cleaning:")
	print(cleaned.isna().sum())

	return cleaned


def run_preprocessing(input_path: Path = RAW_DATA_PATH, output_path: Path = CLEAN_DATA_PATH) -> pd.DataFrame:
	"""Execute the end-to-end preprocessing workflow and persist the cleaned CSV."""

	device_usage = load_dataset(input_path)
	inspect_dataset(device_usage)
	check_data_validity(device_usage)
	cleaned = clean_dataset(device_usage)

	print(f"\nFinal dataset shape: {cleaned.shape}")
	print(cleaned.head())

	output_path.parent.mkdir(parents=True, exist_ok=True)
	cleaned.to_csv(output_path, index=False)
	print(f"Cleaned dataset saved to {output_path}")
	return cleaned


def read_dataset(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
	"""
	Read dataset from CSV file for clustering analysis. 
	
	Args:
		path: Path to the CSV file (defaults to cleaned dataset)
		
	Returns:
		pd.DataFrame: Loaded dataframe
	"""
	if isinstance(path, str):
		path = PROJECT_ROOT / path
	
	return load_dataset(path)


def preprocess_data(df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLS) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Extract features and create a copy for clustering. 
	
	Args:
		df: Input dataframe
		feature_cols: List of feature column names
		
	Returns:
		tuple: (X features, df_clust copy)
	"""
	X = df[feature_cols]
	df_clust = df.loc[X.index].copy()
	
	return X, df_clust


def scale_features(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
	"""
	Standardize features using StandardScaler. 
	(x - mean) / standard_deviation to avoid outliers
	
	Args:
		X: Feature matrix
		
	Returns:
		tuple: (scaled features, fitted scaler)
	"""
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	
	return X_scaled, scaler


def _parse_args() -> tuple[Path, Path]:
	import argparse

	parser = argparse.ArgumentParser(
		description="Load, inspect, validate, and clean the User Behavior dataset."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=RAW_DATA_PATH,
		help="Path to the raw user_behavior_dataset.csv file.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=CLEAN_DATA_PATH,
		help="Destination path for the cleaned CSV.",
	)
	parsed = parser.parse_args()
	return parsed.input, parsed.output


if __name__ == "__main__":
	input_path, output_path = _parse_args()
	run_preprocessing(input_path, output_path)
