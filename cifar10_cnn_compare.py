import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42
TRAIN_RATIO = 0.3
VAL_RATIO = 0.2
EPOCHS = 8
BATCH_SIZE = 64


def to_binary_labels(y):
    y = y.reshape(-1)
    animals = np.isin(y, [2, 3, 4, 5, 6, 7]).astype(np.int32)
    return np.where(animals == 1, 1, 2).astype(np.int32)


def split_data(x, y, train_ratio=TRAIN_RATIO, seed=SEED):
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        cut = int(len(idx) * train_ratio)
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def build_model(conv_layers):
    model = keras.Sequential()
    model.add(layers.Input(shape=(32, 32, 3)))
    filters = [32, 64, 128]
    for i in range(conv_layers):
        model.add(layers.Conv2D(filters[i], (3, 3), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    (x_train_keras, y_train_keras), (x_test_keras, y_test_keras) = keras.datasets.cifar10.load_data()

    x = np.concatenate([x_train_keras, x_test_keras], axis=0).astype("float32") / 255.0
    y = np.concatenate([y_train_keras, y_test_keras], axis=0)
    y = to_binary_labels(y)

    x_train, x_test, y_train, y_test = split_data(x, y)
    y_train_model = y_train - 1
    y_test_model = y_test - 1

    print(f"Trening: {x_train.shape[0]} probek")
    print(f"Test: {x_test.shape[0]} probek")
    print("Klasa 1: zwierzeta")
    print("Klasa 2: pojazdy")

    results = []

    for conv_layers in [1, 2, 3]:
        model = build_model(conv_layers)
        history = model.fit(
            x_train,
            y_train_model,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VAL_RATIO,
            verbose=2,
        )
        test_loss, test_acc = model.evaluate(x_test, y_test_model, verbose=0)
        results.append(
            {
                "model": f"CNN_{conv_layers}",
                "conv_layers": conv_layers,
                "best_val_acc": max(history.history["val_accuracy"]),
                "test_acc": test_acc,
                "test_loss": test_loss,
            }
        )

    results = sorted(results, key=lambda x: x["test_acc"], reverse=True)

    print("\nWyniki:")
    for row in results:
        print(
            f"{row['model']}: warstwy_splotowe={row['conv_layers']}, "
            f"best_val_acc={row['best_val_acc']:.4f}, "
            f"test_acc={row['test_acc']:.4f}, test_loss={row['test_loss']:.4f}"
        )

    best = results[0]
    print(
        f"\nOptymalny klasyfikator: {best['model']} "
        f"(liczba warstw splotowych: {best['conv_layers']}, test_acc={best['test_acc']:.4f})"
    )


if __name__ == "__main__":
    main()
