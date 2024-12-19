import matplotlib.pyplot as plt
import pandas as pd
import skops.io as sio
import mlflow
import mlflow.sklearn
import os
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

# Start MLflow experiment
mlflow.set_experiment("bank_classification_experiment")
mlflow.start_run()

# loading the data (without limiting rows)
try:
    # Attempt to read CSV with 'id' as index column
    bank_df = pd.read_csv("train.csv", index_col="id")
except ValueError:
    # If 'id' column is not available, load the CSV without specifying the index column
    bank_df = pd.read_csv("train.csv")

# Drop unnecessary columns
bank_df = bank_df.drop(["CustomerId", "Surname"], axis=1)

# Optionally shuffle the data (remove if not needed)
bank_df = bank_df.sample(frac=1)  # Random shuffle

# Splitting data into training and testing sets
X = bank_df.drop(["Exited"], axis=1)
y = bank_df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)

# Identify numerical and categorical columns
cat_col = [1, 2]
num_col = [0, 3, 4, 5, 6, 7, 8, 9]

# Transformers for numerical data
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", MinMaxScaler())]
)

# Transformers for categorical data
categorical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder())]
)

# Combine pipelines using ColumnTransformer
preproc_pipe = ColumnTransformer(
    transformers=[("num", numerical_transformer, num_col), ("cat", categorical_transformer, cat_col)],
    remainder="passthrough",
)

# Selecting the best features
KBest = SelectKBest(chi2, k="all")

# Random Forest Classifier with fewer estimators to save memory
model = RandomForestClassifier(n_estimators=10, random_state=125)  # Use fewer estimators for reduced memory usage

# KBest and model pipeline
train_pipe = Pipeline(
    steps=[("KBest", KBest), ("RFmodel", model)],
)

# Combining the preprocessing and training pipelines
complete_pipe = Pipeline(
    steps=[("preprocessor", preproc_pipe), ("train", train_pipe)],
)

# Log model parameters to MLflow
mlflow.log_param("n_estimators", model.n_estimators)
mlflow.log_param("random_state", model.random_state)

# Running the complete pipeline
complete_pipe.fit(X_train, y_train)

# Example input for logging model signature
input_example = X_train.iloc[0].to_dict()  # Convert one row of training data to a dictionary (as input example)

input_example_processed = pd.DataFrame([input_example])  # Convert to DataFrame (2D array)
input_example_processed = preproc_pipe.transform(input_example_processed)  # Apply the same preprocessing

# Log model with signature and input example
mlflow.sklearn.log_model(train_pipe, "model", input_example=input_example_processed)


# Model Evaluation
predictions = complete_pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

# Log metrics to MLflow
mlflow.log_metric("accuracy", accuracy)
mlflow.log_metric("f1_score", f1)

print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, predictions, labels=complete_pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=complete_pipe.classes_)
disp.plot()
plt.savefig("model_results.png", dpi=120)

# Log the confusion matrix plot as artifact
if os.path.exists("model_results.png"):
    mlflow.log_artifact("model_results.png")
else:
    print("Artifact model_results.png not found!")
# Write metrics to file (also as artifact)
with open("metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}\n\n")

# Log metrics file as artifact
if os.path.exists("metrics.txt"):
    mlflow.log_artifact("metrics.txt")
else:
    print("Artifact metrics.txt not found!")

# Saving the pipeline
sio.dump(complete_pipe, "bank_pipeline.skops")
mlflow.log_artifact("bank_pipeline.skops")
if os.path.exists("bank_pipeline.skops"):
    mlflow.log_artifact("bank_pipeline.skops")
else:
    print("Artifact bank_pipeline.skops not found!")

# Log the model pipeline with MLflow
mlflow.sklearn.log_model(complete_pipe, "model")

# End the MLflow run
mlflow.end_run()

# Free up memory
del X_train, y_train, X_test, y_test, complete_pipe
