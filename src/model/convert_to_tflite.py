from pathlib import Path

import tensorflow as tf
import keras

input_path = Path("models/r15_diagonal_rb_1_3_e_100000_lr_0_001_timeout_scaling_battle_big_aggresive_v0_86_snapshot.keras")
output_path = Path("assets/models/model.tflite")

keras_model = keras.models.load_model(input_path)
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

with open(output_path, "wb") as f:
    f.write(tflite_quant_model)

