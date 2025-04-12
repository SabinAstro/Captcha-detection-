from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def build_siamese_model():
    def initialize_base_network():
        input = Input((105, 105, 1))
        x = Conv2D(64, (10, 10), activation='relu')(input)
        x = MaxPooling2D()(x)
        x = Conv2D(128, (7, 7), activation='relu')(x)
        x = MaxPooling2D()(x)
        x = Conv2D(128, (4, 4), activation='relu')(x)
        x = MaxPooling2D()(x)
        x = Conv2D(256, (4, 4), activation='relu')(x)
        x = Flatten()(x)
        x = Dense(4096, activation='sigmoid')(x)
        return Model(input, x)

    def euclidean_distance(vectors):
        x, y = vectors
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    input_a = Input((105, 105, 1))
    input_b = Input((105, 105, 1))

    base_network = initialize_base_network()
    feats_a = base_network(input_a)
    feats_b = base_network(input_b)

    distance = Lambda(euclidean_distance)([feats_a, feats_b])
    model = Model(inputs=[input_a, input_b], outputs=distance)

    return model


def load_data(data_path):
    pairs = []
    labels = []
    writers = os.listdir(data_path)

    for writer in writers:
        genuine_dir = os.path.join(data_path, writer, "genuine")
        forged_dir = os.path.join(data_path, writer, "forged")

        genuine = [os.path.join(genuine_dir, f) for f in os.listdir(genuine_dir)]
        forged = [os.path.join(forged_dir, f) for f in os.listdir(forged_dir)]

        for i in range(len(genuine)):
            for j in range(i+1, len(genuine)):
                pairs.append([genuine[i], genuine[j]])
                labels.append(0)

            for k in range(min(len(forged), 5)):
                pairs.append([genuine[i], forged[k]])
                labels.append(1)

    return pairs, labels


def preprocess_pair(pair):
    def preprocess(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (105, 105))
        img = img.astype("float32") / 255.0
        return img.reshape(105, 105, 1)

    return preprocess(pair[0]), preprocess(pair[1])


def train(data_path):
    pairs, labels = load_data(data_path)
    data_a, data_b = [], []
    for p in pairs:
        img1, img2 = preprocess_pair(p)
        data_a.append(img1)
        data_b.append(img2)

    X1 = np.array(data_a)
    X2 = np.array(data_b)
    y = np.array(labels)

    model = build_siamese_model()
    model.compile(loss="binary_crossentropy", optimizer=Adam(0.0001), metrics=["accuracy"])
    model.fit([X1, X2], y, batch_size=16, epochs=10)
    model.save("model/siamese_model.h5")
