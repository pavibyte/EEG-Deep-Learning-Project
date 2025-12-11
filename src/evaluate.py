# src/evaluate.py
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(model_path=r"C:\Users\VICTUS\OneDrive\Desktop\FINAL YEAR PROJECT\EEG Deep Learning project\saved_models\eeg_dnn_best.h5", scaler_path=r"C:\Users\VICTUS\OneDrive\Desktop\FINAL YEAR PROJECT\EEG Deep Learning project\saved_models\scaler.joblib"):
    import os
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    import pandas as pd
    df = pd.read_csv(r"C:\Users\VICTUS\OneDrive\Desktop\FINAL YEAR PROJECT\EEG Deep Learning project\data\mental-state.csv")
    X = df.drop("Label", axis=1).values
    y = df["Label"].values.astype(int)

    # use same split as preprocess; quick approach: load pre-split saved arrays (if you saved them).
    # Simpler: we will re-split deterministically here.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled).argmax(axis=1)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)

    os.makedirs("saved_models", exist_ok=True)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion matrix")
    plt.savefig("saved_models/confusion_matrix.png")
    print("Saved confusion matrix")
    return y_test, preds

if __name__ == "__main__":
    evaluate()
