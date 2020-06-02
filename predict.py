import cv2
import numpy as np
import tensorflow as tf


def prepare(path):
    IMG_SIZE = 52
    img = cv2.imread(path, 0)
    new_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    new_img = new_img.astype(np.float64)
    return new_img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("model.h5")

prediction = model.predict([prepare("smokey.jpg")])
print(prediction)
