import os
import joblib

from src.data_loader import (
    load_iris_data,
    display_dataset_info
)

from src.preprocessing import prepare_data

from src.models import (
    create_models,
    train_models
)

from src.evaluation import (
    evaluate_models,
    find_best_model
)

from src.visualization import (
    create_visualizations
)

from src.prediction import (
    load_saved_model,
    display_prediction
)


def save_best_model(
    best_model,
    scaler
):
    """
    Save the selected best model and scaler
    into the models folder.
    """

    os.makedirs("models", exist_ok=True)

    model_path = os.path.join(
        "models",
        "best_model.pkl"
    )

    scaler_path = os.path.join(
        "models",
        "scaler.pkl"
    )

    joblib.dump(
        best_model,
        model_path
    )

    joblib.dump(
        scaler,
        scaler_path
    )

    print("\nBest model and scaler saved successfully.")
    print(f"Model saved at  : {model_path}")
    print(f"Scaler saved at : {scaler_path}")


def main():

    print("\n" + "=" * 60)
    print("IRIS DATA CLASSIFICATION USING AI")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1 - LOAD DATASET
    # ---------------------------------------------------------

    print("\n[1] Loading Iris dataset...")
    
    df, iris = load_iris_data()

    print("Dataset loaded successfully.")

    display_dataset_info(df)

    # ---------------------------------------------------------
    # STEP 2 - PREPARE DATA
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("[2] Preparing data...")
    print("=" * 60)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = prepare_data(df)

    print("\nData split completed.")

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Testing samples  : {len(X_test)}"
    )

    print("\nFeature scaling completed.")

    # ---------------------------------------------------------
    # STEP 3 - CREATE MODELS
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("[3] Creating classification models...")
    print("=" * 60)

    models = create_models()

    for name in models:
        print(f"- {name}")

    # ---------------------------------------------------------
    # STEP 4 - TRAIN MODELS
    # ---------------------------------------------------------

    trained_models = train_models(
        models,
        X_train_scaled,
        y_train
    )

    # ---------------------------------------------------------
    # STEP 5 - EVALUATE MODELS
    # ---------------------------------------------------------

    results = evaluate_models(
        trained_models,
        X_test_scaled,
        y_test
    )

    # ---------------------------------------------------------
    # STEP 6 - CREATE VISUALIZATIONS
    # ---------------------------------------------------------

    create_visualizations(results)

    # ---------------------------------------------------------
    # STEP 7 - SELECT BEST MODEL
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("[7] Selecting best model...")
    print("=" * 60)

    best_model_name, best_f1_score = find_best_model(
        results
    )

    best_model = trained_models[
        best_model_name
    ]

    best_accuracy = results[
        best_model_name
    ]["accuracy"]

    print(
        f"\nBest model: {best_model_name}"
    )

    print(
        f"Best Accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        f"Best F1-Score: "
        f"{best_f1_score * 100:.2f}%"
    )

    # Save selected model and scaler
    save_best_model(
        best_model,
        scaler
    )

    # ---------------------------------------------------------
    # STEP 8 - TEST COMPLETELY NEW FLOWER
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("[8] Testing a completely new flower...")
    print("=" * 60)

    try:

        saved_model, saved_scaler = load_saved_model()

        display_prediction(
            saved_model,
            saved_scaler,
            iris
        )

    except FileNotFoundError as error:

        print("\nModel loading error:")
        print(error)

    except Exception as error:

        print("\nPrediction error:")
        print(error)

    # ---------------------------------------------------------
    # PROJECT SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nYour AI classification system has:")

    print("✓ Loaded the Iris dataset")
    print("✓ Analyzed the dataset")
    print("✓ Split data into training and testing sets")
    print("✓ Applied feature scaling")
    print("✓ Trained KNN")
    print("✓ Trained Logistic Regression")
    print("✓ Trained Decision Tree")
    print("✓ Compared model performance")
    print("✓ Generated confusion matrices")
    print("✓ Generated model comparison chart")
    print("✓ Selected the best model automatically")
    print("✓ Saved the best model and scaler")
    print("✓ Tested completely new flower data")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()