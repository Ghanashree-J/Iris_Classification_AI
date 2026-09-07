from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def create_models():
    """
    Create all classification models
    that will be compared.
    """

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),

        "Logistic Regression": LogisticRegression(
            max_iter=200
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        )
    }

    return models


def train_models(models, X_train, y_train):
    """
    Train every model using the training data.
    """

    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

        print(f"{name} training completed.")

    return trained_models