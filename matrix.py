from typing import List

import util


class Matrix:
    def __init__(self, matrix: List[List[float]]):
        assert all(len(row) == len(matrix[0]) for row in matrix)
        self._matrix = matrix

    @staticmethod
    def filled_with(rows: int, cols: int, value: float = 0):
        return Matrix([[value for _ in range(cols)] for _ in range(rows)])

    @staticmethod
    def random(rows: int, cols: int):
        return Matrix([[util.random() for _ in range(cols)] for _ in range(rows)])

    def __getitem__(self, item: (int, int)) -> float:
        return self._matrix[item[0]][item[1]]

    def rows(self) -> int:
        return len(self._matrix)

    def cols(self) -> int:
        return 0 if not self._matrix else len(self._matrix[0])

    def __repr__(self) -> str:
        if not self._matrix:
            return "Matrix([])"
        out = "Matrix([\n"
        out += ",\n".join(("\t[" + ", ".join(map(str, row)) + "]") for row in self._matrix)
        return out + "\n])"

    def __add__(self, other):
        if isinstance(other, Matrix):
            assert self.rows() == other.rows() and self.cols() == other.cols()
        return Matrix([
            [
                self[row, col] + (other[row, col] if isinstance(other, Matrix) else other)
                for col in range(self.cols())
            ]
            for row in range(self.rows())
        ])

    def __sub__(self, other):
        if isinstance(other, Matrix):
            assert self.rows() == other.rows() and self.cols() == other.cols()
        return Matrix([
            [
                self[row, col] - (other[row, col] if isinstance(other, Matrix) else other)
                for col in range(self.cols())
            ]
            for row in range(self.rows())
        ])

    def __mul__(self, other):
        if isinstance(other, Matrix):
            assert self.rows() == other.rows() and self.cols() == other.cols()
        return Matrix([
            [
                self[row, col] * (other[row, col] if isinstance(other, Matrix) else other)
                for col in range(self.cols())
            ]
            for row in range(self.rows())
        ])

    def dot(self, other):
        assert isinstance(other, Matrix)
        assert self.cols() == other.rows()
        return Matrix([
            [
                sum(self[row, i] * other[i, col] for i in range(self.cols()))
                for col in range(other.cols())
            ]
            for row in range(self.rows())
        ])

    def apply_function(self, func):
        return Matrix([
            [
                func(self[row, col])
                for col in range(self.cols())
            ]
            for row in range(self.rows())
        ])

    def transpose(self):
        return Matrix([list(x) for x in zip(*self._matrix)])
