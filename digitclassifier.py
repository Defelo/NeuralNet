from typing import List

import mnist
from matrix import Matrix
from network import Network
from trainingdata import TrainingData
from util import sigmoid, sigmoid_prime


class DigitClassifier(Network):
    def __init__(self, weights: List[Matrix], biases: List[Matrix]):
        super().__init__(weights, biases)

    @classmethod
    def random_network(cls, layer_sizes: List[int]):
        return super().random_network([28 ** 2] + layer_sizes + [10])

    @staticmethod
    def activation(x: float) -> float:
        return sigmoid(x)

    @staticmethod
    def activation_derivative(x: float) -> float:
        return sigmoid_prime(x)

    def prepare_expected_output(self, expected) -> Matrix:
        return Matrix.from_vector([float(i == expected) for i in range(10)])

    def prepare_input(self, inp) -> Matrix:
        return Matrix.from_vector(inp)

    def interpret_output(self, out: Matrix):
        out = out.to_vector()
        return out.index(max(out))

    def evaluate(self, test_data: List[TrainingData]) -> float:
        result = 0
        for td in test_data:
            if self.feedforward(td.input_data) == td.expected_output:
                result += 1
        return result / len(test_data)


if __name__ == '__main__':
    network = DigitClassifier.random_network([10, 10])
    print("Loading training data ...")
    training_data = [x for _, x in zip(range(2000), mnist.load_train())]
    print("Loading validation data ...")
    validation_data = [x for _, x in zip(range(1000), mnist.load_test())]
    print("Start training")
    network.train(training_data, 30, 10, 3, validation_data, "network.json")
    print(f"Accuracy: {network.evaluate(validation_data)}")
