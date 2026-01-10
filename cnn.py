import numpy as np
import pandas as pd
import glob, re
from collections import OrderedDict

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# --- 1) ラベル読み込み（記事：train_master.tsvを想定） ---
train_Y = pd.read_csv("train_master.tsv", delimiter="\t")
train_Y = train_Y.drop("file_name", axis=1)
Y = to_categorical(train_Y)  # (N,10)

# --- 2) 画像ファイルの順序を揃える（記事のsortedStringList相当） ---
def sorted_string_list(paths):
    sort_dict = OrderedDict()
    for p in paths:
        sort_dict[p] = [int(x) for x in re.split(r"(\d+)", p) if x.isdigit()]
    return [k for k, _ in sorted(sort_dict.items(), key=lambda x: x[1])]

train_files = glob.glob("train_images/t*")  # 記事では train_images/train_0.jpg など
train_files = sorted_string_list(train_files)

# --- 3) 画像をnumpy化 + 正規化 ---
X = []
for fp in train_files:
    img = load_img(fp)              # 必要なら target_size=(96,96) などで固定
    arr = img_to_array(img) / 255.0 # 0-1正規化
    X.append(arr)
X = np.array(X, dtype=np.float32)

# --- 4) train/valid/test 分割（記事と同様に sklearn を使ってOK） ---
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

# --- 5) データ拡張（記事：回転+左右反転） ---
train_gen = ImageDataGenerator(rotation_range=45, horizontal_flip=True)
valid_gen = ImageDataGenerator()

train_data = train_gen.flow(X_train, Y_train, batch_size=32, shuffle=False)
valid_data = valid_gen.flow(X_valid, Y_valid, batch_size=32)

# --- 6) CNNモデル（記事後半の「少し複雑版」に寄せる） ---
model = models.Sequential([
    layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=X_train.shape[1:]),
    layers.MaxPooling2D(),
    layers.Dropout(0.5),

    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Dropout(0.5),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=30, restore_best_weights=True
)

history = model.fit(
    train_data,
    epochs=5000,
    validation_data=valid_data,
    callbacks=[early_stop],
)

# --- 7) 評価（記事：classification_report） ---
from sklearn.metrics import classification_report
y_true = np.argmax(Y_test, axis=1)
y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)
print(classification_report(y_true, y_pred))

