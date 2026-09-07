import os
import joblib
import pandas as pd


def load_saved_model():
    """
    Load the best trained model and scaler
    from the models folder.
    """

    model_path = os.path.join(
        "models",
        "best_model.pkl"
    )

    scaler_path = os.path.join(
        "models",
        "scaler.pkl"
    )

    best_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return best_model, scaler


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

    try:
        sepal_length = float(
            input("\nEnter sepal length (cm): ")
        )

        sepal_width = float(
            input("Enter sepal width (cm): ")
        )

        petal_length = float(
            input("Enter petal length (cm): ")
        )

        petal_width = float(
            input("Enter petal width (cm): ")
        )

    except ValueError:
        print("\nInvalid input.")
        print("Please enter numbers only.")
        return

    # Keep feature names consistent with training
    feature_names = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]

    # Create a DataFrame so the scaler
    # receives the same feature names used during training
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

    predicted_class = prediction[0]

# Calculate prediction confidence
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(new_flower_scaled)
        confidence = max(probabilities[0]) * 100
    else:
        confidence = None

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
    print(f"Sepal length : {sepal_length} cm")
    print(f"Sepal width  : {sepal_width} cm")
    print(f"Petal length : {petal_length} cm")
    print(f"Petal width  : {petal_width} cm")

    print("\nPredicted species:")
    print(f">>> {predicted_species}")

    if confidence is not None:
        print(f"AI confidence: {confidence:.2f}%") 

    print("=" * 60)