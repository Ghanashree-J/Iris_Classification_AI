# Iris Flower Classification Using Machine Learning

## About the Project

I made this project to understand how a machine learning classification model works.

For this project, I used the Iris dataset. The dataset contains measurements of flowers such as sepal length, sepal width, petal length and petal width.

Using these measurements, the model learns to identify which type of Iris flower it belongs to.

The three flower types in the dataset are Setosa, Versicolor and Virginica.

## What I Did in This Project

First, I loaded and checked the Iris dataset to understand the number of samples, features and classes.

Then I divided the data into training and testing data. I used 80% of the data for training and 20% for testing.

Since the measurements have different ranges, I also applied feature scaling before training the models.

For the classification part, I used three different machine learning algorithms:

- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree

KNN is the main classification algorithm used in the original project. I added the other two models so that I could compare their performance and understand which model works better for this dataset.

## Model Evaluation

After training the models, I compared them using:

- Accuracy
- Precision
- Recall
- F1-Score

I also generated confusion matrices to see which flower classes were predicted correctly and where the models made mistakes.

A model comparison graph is also created and saved in the `outputs` folder.

## Selecting the Best Model

After comparing the results, the model with the best performance is selected automatically.

The selected model and the scaler are saved using Joblib. This means I can use the trained model later without training it again from the beginning.

## Testing a New Flower

I also tested the system with measurements that were not directly taken from the test set.

For example:

- Sepal length: 5.1 cm
- Sepal width: 3.5 cm
- Petal length: 1.4 cm
- Petal width: 0.2 cm

The system predicted the flower as **Setosa** with an AI confidence of around **98%**.

This part helped me understand how a trained classification model can be used on new data.

## Project Structure

```text
iris_classification_ai/

├── data/
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── outputs/
│   ├── knn_confusion_matrix.png
│   ├── logistic_regression_confusion_matrix.png
│   ├── decision_tree_confusion_matrix.png
│   └── model_comparison.png
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── prediction.py
│
├── main.py
├── requirements.txt
└── README.md