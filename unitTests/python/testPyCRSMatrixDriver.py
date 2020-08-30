"""Tests for the CRSMatrix python wrapper."""

import unittest

import numpy as np
from numpy import testing

from testPyCRSMatrix import get_matrix_double, get_matrix_int


def clear(matrix):
    for row in range(matrix.num_rows()):
        cols = np.array(matrix.get_columns(row))
        matrix.remove_nonzeros(row, cols)
        matrix.compress()


class CRSMatrixTests(unittest.TestCase):

    def setUp(self):
        for matrix in (get_matrix_double(True), get_matrix_int(True)):
            clear(matrix)

    def test_resize(self):
        for matrix in (get_matrix_double(True), get_matrix_int(True)):
            for rowcount in range(1, 11):
                for colcount in range(1, 11):
                    matrix.resize(rowcount, colcount)
                    self.assertEqual(matrix.num_rows(), rowcount)
                    self.assertEqual(matrix.num_columns(), colcount)

    def test_insertion(self):
        for matrix in (get_matrix_double(True), get_matrix_int(True)):
            matrix.resize(10, 10)
            dtype = matrix.to_scipy().diagonal().dtype.type
            for initializer in range(7, 17):
                clear(matrix)
                matrix.resize(10, 10)
                for index in range(matrix.num_rows()):
                    matrix.insert_nonzeros(index, np.int16(index), dtype(initializer))
                matrix.compress()
                diagonal = matrix.to_scipy().diagonal()
                expected_diagonal = np.array([initializer for _ in range(10)], dtype=diagonal.dtype)
                testing.assert_array_equal(diagonal, expected_diagonal)

    def test_to_scipy(self):
        for matrix in (get_matrix_double(True), get_matrix_int(True)):
            matrix.resize(0, 0)
            with self.assertRaisesRegex(RuntimeError, "Empty matrices"):
                matrix.to_scipy()
            matrix.resize(5, 5)
            matrix.insert_nonzeros(1, np.int16(1), matrix.to_scipy().dtype.type(1))
            with self.assertRaisesRegex(RuntimeError, "Uncompressed matrices"):
                matrix.to_scipy()

    def test_addtorow(self):
        for matrix in (get_matrix_double(True), get_matrix_int(True)):
            matrix.resize(7, 7)
            dtype = matrix.to_scipy().dtype.type
            for row in range(matrix.num_rows()):
                matrix.insert_nonzeros(row, np.array(range(matrix.num_columns()), dtype=np.int16), np.array(range(matrix.num_columns()), dtype=dtype))
                matrix.add_to_row(row, np.array(range(matrix.num_columns()), dtype=np.int16), np.array(range(matrix.num_columns()), dtype=dtype))
                testing.assert_array_equal(matrix.to_scipy().toarray()[0], 2 * np.array(range(matrix.num_columns()), dtype=dtype))

if __name__ == '__main__':
    unittest.main()
