from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def create_models():
    """
    Create the classification models used in the project.

    KNN is the main algorithm required by the training project.
    Logistic Regression and Decision Tree are added for comparison.
    """

    models = {
        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        )
    }

    return models


def train_models(
    models,
    X_train_scaled,
    y_train
):
    """
    Train all classification models.

    Returns:
        Dictionary containing trained models.
    """

    trained_models = {}

    print("\n" + "=" * 60)
    print("TRAINING CLASSIFICATION MODELS")
    print("=" * 60)

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(
            X_train_scaled,
            y_train
        )

        trained_models[name] = model

        print(f"{name} training completed.")

    return trained_models