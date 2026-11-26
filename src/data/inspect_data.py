import pandas as pd
from pathlib import Path

DATA_PATH = Path(
    "data/raw/telco_customer_churn.csv"
)

def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )
    return pd.read_csv(path)

def inspect_dataset(df: pd.DataFrame):
    print("\n--- DATASET SHAPE ---")
    print(df.shape)

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- DATASET INFO ---")
    df.info()

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    print("\n--- DUPLICATE ROWS ---")
    print(df.duplicated().sum())

    print("\n--- UNIQUE VALUES ---")
    print(df.nunique())

    print("\n--- NUMERICAL SUMMARY ---")
    print(df.describe())

    print("\n--- CATEGORICAL SUMMARY ---")
    print(df.describe(include="object"))

    print("\n--- TARGET DISTRIBUTION ---")
    print(df["Churn"].value_counts(dropna=False))

if __name__ == "__main__":

    df = load_dataset(DATA_PATH)

    inspect_dataset(df)