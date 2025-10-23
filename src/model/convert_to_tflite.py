import sys
from pathlib import Path

import tensorflow as tf
import keras

input_path = sys.argv[1]
output_path = sys.argv[2]

keras_model = keras.models.load_model(input_path)
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

with open(output_path, "wb") as f:
    f.write(tflite_quant_model)

