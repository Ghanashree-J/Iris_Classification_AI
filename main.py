import sys
import os
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_iris_data, display_dataset_info
from preprocessing import prepare_data
from models import create_models, train_models
from evaluation import evaluate_all_models, display_results
from visualization import (
    plot_all_confusion_matrices,
    plot_model_comparison
)
from prediction import (
    display_prediction,
    load_saved_model
)


def main():

    print("\n" + "=" * 70)
    print("        IRIS DATA CLASSIFICATION USING AI")
    print("=" * 70)

    # ---------------------------------------------------------
    # STEP 1: LOAD DATASET
    # ---------------------------------------------------------

    print("\n[1] Loading Iris dataset...")

    df, iris = load_iris_data()

    print("Dataset loaded successfully.")

    # ---------------------------------------------------------
    # STEP 2: UNDERSTAND DATASET
    # ---------------------------------------------------------

    display_dataset_info(df)

    # ---------------------------------------------------------
    # STEP 3: PREPARE DATA
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("[2] Preparing data...")
    print("=" * 70)

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

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    print("\nFeature scaling completed.")

    # ---------------------------------------------------------
    # STEP 4: CREATE MODELS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("[3] Creating classification models...")
    print("=" * 70)

    models = create_models()

    for model_name in models:
        print(f" - {model_name}")

    # ---------------------------------------------------------
    # STEP 5: TRAIN MODELS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("[4] Training models...")
    print("=" * 70)

    trained_models = train_models(
        models,
        X_train_scaled,
        y_train
    )

    # ---------------------------------------------------------
    # STEP 6: EVALUATE MODELS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("[5] Evaluating models...")
    print("=" * 70)

    results = evaluate_all_models(
        trained_models,
        X_test_scaled,
        y_test
    )

    # ---------------------------------------------------------
    # STEP 7: DISPLAY RESULTS
    # ---------------------------------------------------------

    display_results(results)

    # ---------------------------------------------------------
    # STEP 8: CREATE VISUALIZATIONS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("[6] Creating visualizations...")
    print("=" * 70)

    plot_all_confusion_matrices(results)

    plot_model_comparison(results)

    print("\nAll visualizations have been saved")
    print("inside the 'outputs' folder.")

    # ---------------------------------------------------------
    # STEP 9: FIND BEST MODEL
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("[7] Selecting best model...")
    print("=" * 70)

    best_model_name = max(
        results,
        key=lambda name: (
            results[name]["accuracy"],
            results[name]["f1_score"]
    )
)

    
    best_model = trained_models[best_model_name]

    best_f1 = results[best_model_name]["f1_score"]

    # ---------------------------------------------------------
    # LOAD SAVED BEST MODEL
    # ---------------------------------------------------------

    saved_model, saved_scaler = load_saved_model()

    print("\nSaved model loaded successfully.")
    print("Saved scaler loaded successfully.")

    # Use the saved model and scaler for prediction
    best_model = saved_model
    scaler = saved_scaler

    print(f"\nBest model: {best_model_name}")
    print(f"Best Accuracy: {results[best_model_name]['accuracy'] * 100:.2f}%")
    print(f"Best F1-Score: {best_f1 * 100:.2f}%")

     # ---------------------------------------------------------
    # SAVE BEST MODEL AND SCALER
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # STEP 10: TEST COMPLETELY NEW DATA
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("[8] Testing a completely new flower...")
    print("=" * 70)

    # Example new flower measurements
    sepal_length = 5.8
    sepal_width = 2.7
    petal_length = 4.1
    petal_width = 1.0

 # ---------------------------------------------------------
# TEST A COMPLETELY NEW FLOWER
# ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("[8] Testing a completely new flower...")
    print("=" * 70)

    display_prediction(
        best_model,
        scaler,
        iris 
)

    # ---------------------------------------------------------
    # PROJECT COMPLETED
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 70)

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
    print("✓ Selected the best model")
    print("✓ Tested completely new flower data")


if __name__ == "__main__":
    main()