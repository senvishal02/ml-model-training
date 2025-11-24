import tensorflow as tf
import os
import numpy as np

def train():
    # tiny dummy training
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28,28,1)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation="relu"),
      tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=1, batch_size=128)

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "model.h5"))

if __name__ == "__main__":
    train()
