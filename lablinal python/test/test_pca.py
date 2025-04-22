import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from random import randint, uniform
from pca import *

class TestGaussSolver(unittest.TestCase):
    def test_gauss_solver_unique_solution(self):
        A = Matrix(num_rows=2, num_cols=2, arr=[[2, 3], [1, 1]])
        B = Matrix(num_rows=2, num_cols=1, arr=[[4], [2]])
        solutions = gauss_solver(A, B, eps=1e-8)
        self.assertEqual(len(solutions), 1) # Единственное решение
        sol = solutions[0]
        # sol это (2x1) матрица-столбец
        x1 = sol[1, 1]
        x2 = sol[2, 1]
        self.assertAlmostEqual(x1, 2.0, places=3)
        self.assertAlmostEqual(x2, 0.0, places=3)

    def test_gauss_solver_no_solution(self):
        A = Matrix(num_rows=2, num_cols=2, arr=[[1, 1],[1, 1]])
        B = Matrix(num_rows=2, num_cols=1, arr=[[1], [2]])
        with self.assertRaises(ValueError):
            gauss_solver(A,B)

    def test_gauss_solver_infinite_solutions(self):
        A = Matrix(num_rows=2, num_cols=2, arr=[[1, 1], [2, 2]])
        B = Matrix(num_rows=2, num_cols=1, arr=[[2], [4]])
        solutions = gauss_solver(A, B)
        self.assertTrue(len(solutions) >= 1)
        # Проверим, что среди решений есть хотя бы два линейно независимых (разных) вектора
        if len(solutions) > 1:
            unique = set(tuple(round(solutions[i][r+1,1], 6) for r in range(2)) for i in range(len(solutions)))
            self.assertTrue(len(unique) > 1)
        # Проверим, что хотя бы одно решение действительно удовлетворяет системе (только для частного решения)

        x1 = solutions[0][1, 1]
        x2 = solutions[0][2, 1]
        self.assertAlmostEqual(A[1,1]*x1 + A[1,2]*x2, B[1,1], places=5)
        self.assertAlmostEqual(A[2,1]*x1 + A[2,2]*x2, B[2,1], places=5)

class TestReconstructionError(unittest.TestCase):
    def test_reconstruction_error(self):
        orig = Matrix(num_rows=2, num_cols=2, arr=[[1, 2], [3, 4]])
        recon = Matrix(num_rows=2, num_cols=2, arr=[[1, 2], [3.1, 3.9]])
        err = reconstruction_error(orig, recon)
        # MSE = ((0 + 0) + (0.1^2 + (-0.1)^2)) / 4 = (0.02) / 4= 0.005
        self.assertAlmostEqual(err, 0.005, places=5)

class TestEigenvalues(unittest.TestCase):
    def test_find_eigenvalues_2x2(self):
        arr = [[2, 0],
               [0, 5]]
        M = Matrix(num_rows=2, num_cols=2, arr=arr)
        e = find_eigenvalues(M)
        self.assertAlmostEqual(e[0], 5, places=1)
        self.assertAlmostEqual(e[1], 2, places=1)

    def test_find_eigenvalues_off_diag(self):
        arr = [[4, 1],
               [1, 4]]
        M = Matrix(num_rows=2, num_cols=2, arr=arr)
        e = find_eigenvalues(M)
        self.assertEqual(len(e),2)
        self.assertAlmostEqual(e[0],5, places=1)
        self.assertAlmostEqual(e[1],3, places=1)

class TestApplyPcaToDataset(unittest.TestCase):
    def test_iris_k2(self):
        X_proj, ratio = apply_pca_to_dataset("iris", 2)
        self.assertEqual(X_proj.num_columns,2)
        self.assertTrue(0 <= ratio <= 1)

class TestAutoSelectK(unittest.TestCase):
    def test_auto_select_k(self):
        e = [5, 2, 1, 0.5, 0.1]
        k = auto_select_k(e, threshold=0.8)
        # sum = 8.6, 80% из sum = 6.88
        # 5+2=7 => это > 6.88 => k = 2
        self.assertEqual(k,2)

class TestHandleMissingValues(unittest.TestCase):
    def test_handle_missing_values(self):
        arr = [[1, None, 3], [4, 5, None]]
        X = Matrix(num_rows=2, num_cols=3, arr=arr)
        X_filled = handle_missing_values(X)
        # Не должно быть нанов
        for r in range(1, 3):
            for c in range(1, 4):
                val = X_filled[r, c]
                self.assertIsNotNone(val)

if __name__ == '__main__':
    unittest.main()
