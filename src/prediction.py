from pathlib import Path
import math
import joblib
import pandas as pd


# Get the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"


def load_saved_model():
    """
    Load the best trained model and scaler
    from the models folder.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model not found at: {MODEL_PATH}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found at: {SCALER_PATH}"
        )

    best_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("Saved model loaded successfully.")
    print("Saved scaler loaded successfully.")

    return best_model, scaler


def get_valid_measurement(feature_name):
    """
    Ask the user for a valid positive flower measurement.
    """

    while True:

        try:
            value = float(
                input(f"Enter {feature_name} (cm): ")
            )

            if not math.isfinite(value):
                print("Invalid number. Please enter a normal numeric value.")
                continue

            if value <= 0:
                print("Measurement must be greater than 0.")
                continue

            if value > 10:
                print("Measurement must be 10 cm or less.")
                continue

            return value

        except ValueError:
            print("Invalid input. Please enter numbers only.")


def display_prediction(model, scaler, iris):
    """
    Take flower measurements from the user
    and predict the Iris species.
    """

    print("\n" + "=" * 60)
    print("INTERACTIVE NEW FLOWER PREDICTION")
    print("=" * 60)

    print("\nEnter the flower measurements below.")
    print("Example values: 5.8, 2.7, 4.1, 1.0")

    # Get validated measurements
    sepal_length = get_valid_measurement("sepal length")
    sepal_width = get_valid_measurement("sepal width")
    petal_length = get_valid_measurement("petal length")
    petal_width = get_valid_measurement("petal width")

    # Keep feature names consistent with training
    feature_names = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]

    # Create DataFrame
    new_flower = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=feature_names
    )

    # Apply the saved scaler
    new_flower_scaled = scaler.transform(new_flower)

    # Make prediction
    prediction = model.predict(new_flower_scaled)

    predicted_class = int(prediction[0])

    # Calculate prediction confidence
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            new_flower_scaled
        )

        confidence = max(probabilities[0]) * 100

    # Convert numeric class to species name
    species_names = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    predicted_species = species_names.get(
        predicted_class,
        "Unknown"
    )

    print("\n" + "=" * 60)
    print("NEW FLOWER PREDICTION")
    print("=" * 60)

    print("\nInput measurements:")
    print(f"Sepal length : {sepal_length:.2f} cm")
    print(f"Sepal width  : {sepal_width:.2f} cm")
    print(f"Petal length : {petal_length:.2f} cm")
    print(f"Petal width  : {petal_width:.2f} cm")

    print("\nPredicted species:")
    print(f">>> {predicted_species}")

    if confidence is not None:
        print(f"AI confidence: {confidence:.2f}%")

    print("=" * 60)