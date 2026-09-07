from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(df):
    """
    Separate features and target,
    split the dataset into training and testing sets,
    and scale the features.
    """

    feature_columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]

    # Input features
    X = df[feature_columns]

    # Target
    y = df["target"]

    # 80% Training / 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Feature scaling
    scaler = StandardScaler()

    # Fit scaler only on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the same scaler to testing data
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler
    )