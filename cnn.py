import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

seed = 50
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

out_dir = "output"
os.makedirs(out_dir, exist_ok=True)

(X_train_all, y_train_all), (X_test_all, y_test_all) = cifar10.load_data()

X_train_all = X_train_all.astype("float32") / 255.0
X_test_all  = X_test_all.astype("float32")  / 255.0

y_train_all = to_categorical(y_train_all, 10)
y_test_all  = to_categorical(y_test_all, 10)

def build_model(input_shape, use_bn=False):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    if use_bn: model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))
    if use_bn: model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    if use_bn: model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    if use_bn: model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512))
    if use_bn: model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    opt = keras.optimizers.RMSprop(learning_rate=0.0001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    return model

settings = [
    (300, 100),
    (3000, 1000),
    (50000, 10000),
]

bn_settings = [
    (False, "noBN"),
    (True,  "withBN"),
]

results = []

for use_bn, bn_tag in bn_settings:
    for n_train, n_test in settings:
        print("\n==============================")
        print(f"Run: {bn_tag}  train={n_train}, test={n_test}")
        print("==============================")

        X_train = X_train_all[:n_train]
        y_train = y_train_all[:n_train]
        X_test  = X_test_all[:n_test]
        y_test  = y_test_all[:n_test]

        model = build_model(input_shape=X_train.shape[1:], use_bn=use_bn)

        # 学習
        history = model.fit(
            X_train, y_train,
            batch_size=100,
            epochs=10,
            validation_data=(X_test, y_test),
            verbose=1
        )

        # 評価
        scores = model.evaluate(X_test, y_test, verbose=1)
        test_loss = float(scores[0])
        test_acc  = float(scores[1])

        results.append({
            "bn": bn_tag,
            "train_size": n_train,
            "test_size": n_test,
            "test_loss": test_loss,
            "test_accuracy": test_acc
        })

        plt.figure(figsize=(12, 4))

        # accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend()
        plt.title(f'Accuracy ({bn_tag}, train={n_train}, test={n_test})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        # loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.title(f'Loss ({bn_tag}, train={n_train}, test={n_test})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')

        plt.tight_layout()

        plot_path = os.path.join(out_dir, f"training_curve_{bn_tag}_train{n_train}_test{n_test}_seed{seed}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"Saved plot: {plot_path}")
        print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")

csv_path = os.path.join(out_dir, f"results_bn_compare_seed{seed}.csv")
with open(csv_path, "w", encoding="utf-8") as f:
    f.write("bn,train_size,test_size,test_loss,test_accuracy\n")
    for r in results:
        f.write(f"{r['bn']},{r['train_size']},{r['test_size']},{r['test_loss']},{r['test_accuracy']}\n")

print("Results:")
for r in results:
    print(r)

