# src/plot_training_curves.py
import joblib
import matplotlib.pyplot as plt
import os

def plot_curves(history_path="saved_models/train_history.joblib"):
    history = joblib.load(history_path)

    acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    loss = history["loss"]
    val_loss = history["val_loss"]

    os.makedirs("saved_models", exist_ok=True)

    # Accuracy plot
    plt.figure()
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("saved_models/accuracy_curve.png")
    plt.close()

    # Loss plot
    plt.figure()
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("saved_models/loss_curve.png")
    plt.close()

    print("Training curves saved in saved_models/")

if __name__ == "__main__":
    plot_curves()
