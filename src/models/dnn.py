# src/models/dnn.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_dnn(input_dim, num_classes,
              dropout_rates=(0.35,0.25,0.15),
              units=(512,256,128),
              lr=1e-3):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    model.add(layers.Dense(units[0], activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rates[0]))

    model.add(layers.Dense(units[1], activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rates[1]))

    model.add(layers.Dense(units[2], activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dropout_rates[2]))

    model.add(layers.Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("DNN builder ready")
