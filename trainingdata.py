from matrix import Matrix


class TrainingData:
    def __init__(self, input_data: Matrix, expected_output: Matrix):
        self.input_data = input_data
        self.expected_output = expected_output
