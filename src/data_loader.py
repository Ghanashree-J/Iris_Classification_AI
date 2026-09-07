from sklearn.datasets import load_iris
import pandas as pd


def load_iris_data():
    """
    Load the Iris dataset and return it as a Pandas DataFrame.
    """

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["target"] = iris.target

    df["species"] = df["target"].map({
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    })

    return df, iris


def display_dataset_info(df):
    """
    Display basic information about the dataset.
    """

    print("\n" + "=" * 60)
    print("IRIS DATASET INFORMATION")
    print("=" * 60)

    print(f"Number of samples: {len(df)}")
    print("Number of features: 4")
    print(f"Number of classes: {df['species'].nunique()}")

    print("\nFeature columns:")

    for column in df.columns[:4]:
        print(f" - {column}")

    print("\nClasses:")

    for species in df["species"].unique():
        print(f" - {species}")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nClass distribution:")
    print(df["species"].value_counts())