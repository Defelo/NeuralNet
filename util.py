import math
import random as _random


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def sigmoid_prime(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


def random():
    return _random.random() * 2 - 1
