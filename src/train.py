# src/train.py
from src.preprocessing.preprocess import preprocess_split
from src.models.dnn import build_dnn
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import joblib
import os

def train_main(do_pca=False, pca_components=200, epochs=60, batch_size=32):
    X_train, X_test, y_train, y_test = preprocess_split(
        do_pca=do_pca, pca_components=pca_components
    )
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = build_dnn(input_dim, num_classes)

    os.makedirs("saved_models", exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
        ModelCheckpoint("saved_models/eeg_dnn_best.h5", monitor="val_loss", save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    model.save("saved_models/eeg_dnn_final.h5")
    # Save history small summary
    joblib.dump(history.history, "saved_models/train_history.joblib")
    print("Training complete. Model saved to saved_models/")
    return model, (X_test, y_test)

if __name__ == "__main__":
    train_main()
