from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained classification model
    using multiple performance metrics.
    """

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Detailed classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Setosa",
            "Versicolor",
            "Virginica"
        ],
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report
    }


def evaluate_all_models(
    trained_models,
    X_test,
    y_test
):
    """
    Evaluate all trained models and
    return their results.
    """

    results = {}

    for name, model in trained_models.items():

        print(f"\nEvaluating {name}...")

        results[name] = evaluate_model(
            model,
            X_test,
            y_test
        )

        print(f"{name} evaluation completed.")

    return results


def display_results(results):
    """
    Display the performance of every model.
    """

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 70)

    for name, metrics in results.items():

        print(f"\n{name}")
        print("-" * 50)

        print(
            f"Accuracy  : {metrics['accuracy'] * 100:.2f}%"
        )

        print(
            f"Precision : {metrics['precision'] * 100:.2f}%"
        )

        print(
            f"Recall    : {metrics['recall'] * 100:.2f}%"
        )

        print(
            f"F1-Score  : {metrics['f1_score'] * 100:.2f}%"
        )

        print("\nClassification Report:")
        print(metrics["classification_report"])

        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])