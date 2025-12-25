# src/preprocessing/preprocess.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

def load_data():
    import os
    import pandas as pd

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
    DATA_PATH = os.path.join(BASE_DIR, "data", "mental-state.csv")

    df = pd.read_csv(DATA_PATH)
    X = df.drop("Label", axis=1)
    y = df["Label"].astype(int)
    return X, y


def preprocess_split(path=r"C:\Users\VICTUS\OneDrive\Desktop\FINAL YEAR PROJECT\EEG Deep Learning project\data\mental-state.csv", test_size=0.2, random_state=42,
                     do_pca=False, pca_components=200, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, f"{save_dir}/scaler.joblib")

    pca = None
    if do_pca:
        pca = PCA(n_components=pca_components, random_state=random_state)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        joblib.dump(pca, f"{save_dir}/pca.joblib")

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = preprocess_split()
    print("X_train shape:", X_tr.shape)
    print("X_test shape:", X_te.shape)
