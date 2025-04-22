import unittest
import numpy as np
from random import randint, uniform
from matrix import Matrix

class TestMatrixOperations(unittest.TestCase):
    def setUp(self):
        """Установка общих данных для тестов"""
        self.small_matrix_1 = Matrix(3, 3, 2, [[-5, -1, 2], [10, 4, 1], [3, 3, 1]])
        self.small_matrix_2 = Matrix(3, 3, 2, [[-1, 0, 0], [0, -3.07, -1], [0, 0, -2]])

    def assertMatrixEqual(self, matrix: Matrix, numpy_array: np.ndarray):
        """Сравнивает матрицу с массивом numpy с учётом точности"""
        mat_list = matrix.list()
        np_array_rounded = np.round(numpy_array, matrix.accuracy)
        self.assertEqual(len(mat_list), np_array_rounded.shape[0])
        self.assertEqual(len(mat_list[0]), np_array_rounded.shape[1])

        for row in range(len(mat_list)):
            for col in range(len(mat_list[row])):
                self.assertLessEqual(abs(mat_list[row][col] - np_array_rounded[row, col]), (10 ** -matrix.accuracy) * 2.1)

    def test_addition(self):
        """Тест сложения матриц"""
        result_matrix = self.small_matrix_1 + self.small_matrix_2
        numpy_result = np.array([[-5, -1, 2], [10, 4, 1], [3, 3, 1]]) + np.array([[-1, 0, 0], [0, -3.07, -1], [0, 0, -2]])
        self.assertMatrixEqual(result_matrix, numpy_result)

    def test_scalar_multiplication(self):
        """Тест умножения матрицы на скаляр"""
        scalar = 2
        result_matrix = self.small_matrix_1 * scalar
        numpy_result = np.array([[-5, -1, 2], [10, 4, 1], [3, 3, 1]]) * scalar
        self.assertMatrixEqual(result_matrix, numpy_result)

    def test_matrix_multiplication(self):
        """Тест умножения двух матриц"""
        result_matrix = self.small_matrix_1 * self.small_matrix_2
        numpy_result = np.dot(np.array([[-5, -1, 2], [10, 4, 1], [3, 3, 1]]),
                              np.array([[-1, 0, 0], [0, -3.07, -1], [0, 0, -2]]))
        self.assertMatrixEqual(result_matrix, numpy_result)

    def test_transpose(self):
        """Тест транспонирования матрицы"""
        result_matrix = self.small_matrix_1.transpose()
        numpy_result = np.array([[-5, -1, 2], [10, 4, 1], [3, 3, 1]]).T
        self.assertMatrixEqual(result_matrix, numpy_result)

    def test_trace(self):
        """Тест вычисления следа матрицы"""
        trace = self.small_matrix_1.get_trace()
        numpy_trace = np.trace(np.array([[-5, -1, 2], [10, 4, 1], [3, 3, 1]]))
        self.assertLessEqual(abs(trace - numpy_trace), 10 ** -self.small_matrix_1.accuracy)

    def test_determinant(self):
        """Тест вычисления определителя"""
        determinant = self.small_matrix_1.get_determinant()
        numpy_determinant = np.linalg.det(np.array([[-5, -1, 2], [10, 4, 1], [3, 3, 1]]))
        self.assertLessEqual(abs(determinant - numpy_determinant), (10 ** -self.small_matrix_1.accuracy) * 2.1)

    def test_large_sparse_matrix(self):
        """Тест с большой разреженной матрицей"""
        size = 1000  # Размерность матрицы
        num_nonzero = 1000  # Количество ненулевых элементов
        sparse_array = [[0] * size for _ in range(size)]

        # Заполняем случайными ненулевыми элементами
        for _ in range(num_nonzero):
            row = randint(0, size - 1)
            col = randint(0, size - 1)
            value = round(uniform(-100, 100), 2)
            sparse_array[row][col] = value

        sparse_matrix = Matrix(size, size, 2, sparse_array)
        numpy_sparse = np.array(sparse_array)

        # Проверяем транспонирование
        transposed_matrix = sparse_matrix.transpose()
        numpy_transposed = numpy_sparse.T
        self.assertMatrixEqual(transposed_matrix, numpy_transposed)

        # Проверяем умножение на скаляр
        scalar = 1.5
        scaled_matrix = sparse_matrix * scalar
        numpy_scaled = numpy_sparse * scalar
        self.assertMatrixEqual(scaled_matrix, numpy_scaled)
