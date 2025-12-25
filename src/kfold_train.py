# src/kfold_train.py
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
from src.models.dnn import build_dnn
import pandas as pd

def load_full_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "mental-state.csv")

    df = pd.read_csv(DATA_PATH)
    X = df.drop("Label", axis=1).values
    y = df["Label"].astype(int).values
    return X, y

def run_kfold(k=5, epochs=40, batch_size=32):
    X, y = load_full_data()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n========== Fold {fold}/{k} ==========")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale inside fold (VERY IMPORTANT)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model = build_dnn(
            input_dim=X_train.shape[1],
            num_classes=len(np.unique(y))
        )

        es = EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        )

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1
        )

        preds = np.argmax(model.predict(X_val), axis=1)
        acc = accuracy_score(y_val, preds)
        fold_accuracies.append(acc)

        print(f"Fold {fold} Accuracy: {acc:.4f}")

    print("\n========== K-FOLD RESULTS ==========")
    print(f"Accuracies: {fold_accuracies}")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Std Deviation: {np.std(fold_accuracies):.4f}")

    return fold_accuracies

if __name__ == "__main__":
    run_kfold(k=5)
