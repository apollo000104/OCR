import os

import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers

characters = ['2', '3', '4', '5', '6', '7', '8', '9', 'b', 'c', 'd', 'f', 'g', 'h', 'm', 'n', 'p', 'q', 'r', 's', 't',
              'v', 'w', 'x', 'y', 'z']

# Desired image dimensions
img_width = 200
img_height = 60

downsample_factor = 4

max_length = 6

char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)

num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def encode_single_sample(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    return img


class CTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


model = load_model('my_model_4.h5', custom_objects={'CTCLayer': CTCLayer})

# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
prediction_model.summary()


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
              :, :max_length
              ]

    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def rename_file(file_path, new_name):
    directory = os.path.dirname(file_path)
    new_file_path = os.path.join(directory, new_name)
    os.rename(file_path, new_file_path)


def print_absolute_file_paths(folder_path):
    for root, dirs, files in sorted(os.walk(folder_path)):
        for file in files:
            try:
                file_path = os.path.join(root, file)
                img = encode_single_sample(file_path)
                img = tf.expand_dims(img, axis=0)
                pred = prediction_model.predict(img)
                pred_texts = decode_batch_predictions(pred)
                rename_file(file_path, str(pred_texts).replace("[", "").replace("]", "").replace("'", "").replace("UNK",
                                                                                                                  "") + ".png")
                print(file_path + " " + str(pred_texts))
            except:
                print("error " + file_path)


print_absolute_file_paths(r'./testimages/')