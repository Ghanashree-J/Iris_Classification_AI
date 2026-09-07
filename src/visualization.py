import os

import matplotlib.pyplot as plt
import seaborn as sns


def create_output_folder():
    """
    Create the outputs folder if it does not exist.
    """

    os.makedirs("outputs", exist_ok=True)


def plot_confusion_matrix(
    confusion_matrix_data,
    model_name
):
    """
    Create and save a confusion matrix
    for a classification model.
    """

    create_output_folder()

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        confusion_matrix_data,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Setosa",
            "Versicolor",
            "Virginica"
        ],
        yticklabels=[
            "Setosa",
            "Versicolor",
            "Virginica"
        ]
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    filepath = os.path.join(
        "outputs",
        filename
    )

    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"Saved: {filepath}")


def plot_all_confusion_matrices(results):
    """
    Create confusion matrices for all models.
    """

    for model_name, metrics in results.items():

        plot_confusion_matrix(
            metrics["confusion_matrix"],
            model_name
        )


def plot_model_comparison(results):
    """
    Create a comparison chart for all models.
    """

    create_output_folder()

    model_names = list(results.keys())

    accuracy = [
        results[name]["accuracy"] * 100
        for name in model_names
    ]

    precision = [
        results[name]["precision"] * 100
        for name in model_names
    ]

    recall = [
        results[name]["recall"] * 100
        for name in model_names
    ]

    f1_scores = [
        results[name]["f1_score"] * 100
        for name in model_names
    ]

    x = range(len(model_names))
    width = 0.2

    plt.figure(figsize=(10, 6))

    plt.bar(
        [i - 1.5 * width for i in x],
        accuracy,
        width,
        label="Accuracy"
    )

    plt.bar(
        [i - 0.5 * width for i in x],
        precision,
        width,
        label="Precision"
    )

    plt.bar(
        [i + 0.5 * width for i in x],
        recall,
        width,
        label="Recall"
    )

    plt.bar(
        [i + 1.5 * width for i in x],
        f1_scores,
        width,
        label="F1-Score"
    )

    plt.xlabel("Models")
    plt.ylabel("Score (%)")
    plt.title("Classification Model Comparison")

    plt.xticks(
        list(x),
        model_names
    )

    plt.ylim(0, 100)
    plt.legend()

    plt.tight_layout()

    filepath = os.path.join(
        "outputs",
        "model_comparison.png"
    )

    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"Saved: {filepath}")