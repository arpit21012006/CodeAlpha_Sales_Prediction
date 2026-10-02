from pathlib import Path
import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "Advertising.csv"

FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"


def load_data():
    """Load the advertising dataset."""
    df = pd.read_csv(DATA_PATH)

    # Remove the unnecessary index column.
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    return df


def clean_data(df):
    """Prepare the dataset for analysis and prediction."""
    df = df.copy()

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    # Convert expected columns to numeric values.
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove duplicate records and incomplete rows.
    df = df.drop_duplicates()
    df = df.dropna(subset=required_columns)

    return df.reset_index(drop=True)


def prepare_features(df):
    """Separate advertising budgets and sales."""
    X = df[FEATURES]
    y = df[TARGET]

    return X, y


def get_dataset():
    """Return the cleaned dataset."""
    return clean_data(load_data())