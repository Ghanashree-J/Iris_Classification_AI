from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def evaluate_models(
    trained_models,
    X_test_scaled,
    y_test
):
    """
    Evaluate all trained classification models.

    Calculates:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - Classification Report
    - Confusion Matrix

    Returns:
        Dictionary containing evaluation results.
    """

    results = {}

    print("\n" + "=" * 60)
    print("EVALUATING CLASSIFICATION MODELS")
    print("=" * 60)

    for name, model in trained_models.items():

        print(f"\nEvaluating {name}...")

        # Make predictions
        predictions = model.predict(X_test_scaled)

        # Calculate performance metrics
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        # Classification report
        report = classification_report(
            y_test,
            predictions,
            target_names=[
                "Setosa",
                "Versicolor",
                "Virginica"
            ],
            zero_division=0
        )

        # Confusion matrix
        matrix = confusion_matrix(
            y_test,
            predictions
        )

        # Store all results
        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "classification_report": report,
            "confusion_matrix": matrix,
            "predictions": predictions
        }

        print(f"{name} evaluation completed.")

    # Display comparison
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 60)

    for name, result in results.items():

        print(f"\n{name}")
        print("-" * 50)

        print(
            f"Accuracy  : "
            f"{result['accuracy'] * 100:.2f}%"
        )

        print(
            f"Precision : "
            f"{result['precision'] * 100:.2f}%"
        )

        print(
            f"Recall    : "
            f"{result['recall'] * 100:.2f}%"
        )

        print(
            f"F1-Score  : "
            f"{result['f1_score'] * 100:.2f}%"
        )

        print("\nClassification Report:")
        print(result["classification_report"])

        print("Confusion Matrix:")
        print(result["confusion_matrix"])

    return results


def find_best_model(results):
    """
    Select the best model based on F1-Score.

    F1-Score is used because it balances
    precision and recall.
    """

    if not results:
        raise ValueError(
            "No model evaluation results available."
        )

    best_name = max(
        results,
        key=lambda name: results[name]["f1_score"]
    )

    best_model_score = results[
        best_name
    ]["f1_score"]

    return best_name, best_model_score