import json
import random as _random
from typing import List

from matrix import Matrix
from trainingdata import TrainingData


class Network:
    def __init__(self, weights: List[Matrix], biases: List[Matrix]):
        assert len(weights) == len(biases)
        self._weights = weights
        self._biases = biases
        self._layer_sizes = []

        self._update_layer_sizes()

    def _update_layer_sizes(self):
        self._layer_sizes = [self._weights[0].cols()]
        for i in range(len(self._weights)):
            assert self._weights[i].rows() == self._biases[i].rows()
            assert self._layer_sizes[i] == self._weights[i].cols()
            assert self._biases[i].cols() == 1
            self._layer_sizes.append(self._weights[i].rows())

    @staticmethod
    def random_network(layer_sizes: List[int]) -> (List[Matrix], List[Matrix]):
        return [
                   Matrix.random(layer_sizes[i + 1], layer_sizes[i])
                   for i in range(len(layer_sizes) - 1)
               ], [
                   Matrix.random(layer_sizes[i + 1], 1)
                   for i in range(len(layer_sizes) - 1)
               ]

    def feedforward(self, input_data):
        input_data = self.prepare_input(input_data)
        assert input_data.cols() == 1 and input_data.rows() == self._layer_sizes[0]
        for w, b in zip(self._weights, self._biases):
            input_data = (w.dot(input_data) + b).apply_function(self.activation)
        return self.interpret_output(input_data)

    def backpropagation(self, training_data: TrainingData) -> (List[Matrix], List[Matrix]):
        delta_w = [Matrix.filled_with(w.rows(), w.cols()) for w in self._weights]
        delta_b = [Matrix.filled_with(b.rows(), 1) for b in self._biases]

        layer = self.prepare_input(training_data.input_data)
        activations = [layer]
        zs = []

        # feed forward
        for w, b in zip(self._weights, self._biases):
            z = w.dot(layer) + b
            zs.append(z)
            layer = z.apply_function(self.activation)
            activations.append(layer)

        # propagate backwards
        expected = self.prepare_expected_output(training_data.expected_output)
        delta = (layer - expected) * zs[-1].apply_function(self.activation_derivative)
        delta_w[-1] = delta.dot(activations[-2].transpose())
        delta_b[-1] = delta
        for l in range(2, len(self._layer_sizes)):
            z = zs[-l]
            sp = z.apply_function(self.activation_derivative)
            delta = self._weights[-l + 1].transpose().dot(delta) * sp
            delta_w[-l] = delta.dot(activations[-l - 1].transpose())
            delta_b[-l] = delta
        return delta_w, delta_b

    def update_mini_batch(self, mini_batch: List[TrainingData], learn_rate: float):
        delta_w = [Matrix.filled_with(w.rows(), w.cols()) for w in self._weights]
        delta_b = [Matrix.filled_with(b.rows(), 1) for b in self._biases]

        for td in mini_batch:
            dw, db = self.backpropagation(td)
            delta_w = [x + y for x, y in zip(delta_w, dw)]
            delta_b = [x + y for x, y in zip(delta_b, db)]

        self._weights = [w - dw * (learn_rate / len(mini_batch)) for w, dw in zip(self._weights, delta_w)]
        self._biases = [b - db * (learn_rate / len(mini_batch)) for b, db in zip(self._biases, delta_b)]

    def train(self, training_data: List[TrainingData], epochs: int, mini_batch_size: int, learn_rate: float,
              validation_data: List[TrainingData] = None):
        for epoch in range(epochs):
            _random.shuffle(training_data)
            for i in range(0, len(training_data), mini_batch_size):
                mini_batch = training_data[i:i + mini_batch_size]
                self.update_mini_batch(mini_batch, learn_rate)
            if validation_data:
                print(f"Epoch {epoch + 1}: {self.evaluate(validation_data) * 100}%")
            else:
                print(f"Epoch {epoch + 1} complete")

    def save_to_file(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump({
                "layer_sizes": self._layer_sizes,
                "weights": [w.get_matrix() for w in self._weights],
                "biases": [b.to_vector() for b in self._biases]
            }, f)
            f.flush()

    @classmethod
    def load_file(cls, filepath: str):
        with open(filepath) as f:
            data = json.load(f)
        out = cls(data["weights"], data["biases"])
        assert out._layer_sizes == data["layer_sizes"], "Invalid layer sizes"
        return out

    # abstract methods
    @staticmethod
    def activation(x: float) -> float:
        pass

    @staticmethod
    def activation_derivative(x: float) -> float:
        pass

    def prepare_expected_output(self, expected) -> Matrix:
        pass

    def prepare_input(self, inp) -> Matrix:
        pass

    def interpret_output(self, out: Matrix):
        pass

    def evaluate(self, test_data: List[TrainingData]) -> float:
        pass
