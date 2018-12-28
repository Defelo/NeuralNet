import math
import random as _random


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def random():
    return _random.random() * 2 - 1
