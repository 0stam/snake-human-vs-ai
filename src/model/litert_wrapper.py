from pathlib import Path

import tensorflow as tf
import numpy as np


class LiteRTWrapper:
    def __init__(self, model_path: Path|str, max_batch_size: int):
        self.interpreter = tf.lite.Interpreter(str(model_path))

        self.set_max_batch_size(max_batch_size)

        self._get_input = self.interpreter.tensor(self.interpreter.get_input_details()[0]["index"])
        self._get_output = self.interpreter.tensor(self.interpreter.get_output_details()[0]["index"])

    def set_max_batch_size(self, max_batch_size: int) -> None:
        self.max_batch_size = max_batch_size

        input_details = self.interpreter.get_input_details()[0]
        self.interpreter.resize_tensor_input(input_details["index"], (max_batch_size, *input_details["shape"][1:]))
        self.interpreter.allocate_tensors()

    def predict(self, x: np.ndarray, *args, **kwargs):
        self._get_input()[:x.shape[0]] = x
        self.interpreter.invoke()

        return self._get_output()[:x.shape[0]]  # Potentially buggy if the result is still stored when the next prediction must happen
