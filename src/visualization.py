
import os
import matplotlib.pyplot as plt
import seaborn as sns


def create_visualizations(results):
    """
    Create and save visualizations for model evaluation.

    Generates:
    1. Confusion matrix for each model
    2. Model performance comparison chart
    """

    # Create outputs folder if it does not exist
    os.makedirs("outputs", exist_ok=True)

    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. CONFUSION MATRICES
    # ---------------------------------------------------------

    for name, result in results.items():

        matrix = result["confusion_matrix"]

        # Create a safe filename
        file_name = (
            name.lower()
            .replace(" ", "_")
        )

        output_path = os.path.join(
            "outputs",
            f"{file_name}_confusion_matrix.png"
        )

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            matrix,
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

        plt.title(
            f"{name} - Confusion Matrix"
        )

        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {output_path}")

    # ---------------------------------------------------------
    # 2. MODEL PERFORMANCE COMPARISON
    # ---------------------------------------------------------

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

    plt.xticks(
        list(x),
        model_names
    )

    plt.ylabel("Score (%)")

    plt.xlabel("Classification Model")

    plt.title(
        "Iris Classification Model Performance Comparison"
    )

    plt.ylim(0, 105)

    plt.legend()

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    comparison_path = os.path.join(
        "outputs",
        "model_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {comparison_path}")

    print(
        "\nAll visualizations have been saved "
        "inside the 'outputs' folder."
    )