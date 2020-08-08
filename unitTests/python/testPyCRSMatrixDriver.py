"""Tests for the CRSMatrix python wrapper."""

import unittest

import numpy as np
from numpy import testing

from testPyCRSMatrix import getMatrixOfDoubles, getMatrixOfInts

class CRSMatrixTests(unittest.TestCase):

    def test_resize(self):
        for matrix in (getMatrixOfDoubles(True), getMatrixOfInts(True)):
            for rowcount in range(1, 11):
                for colcount in range(1, 11):
                    matrix.resize(rowcount, colcount)
                    self.assertEqual(matrix.numRows(), rowcount)
                    self.assertEqual(matrix.numColumns(), colcount)

    @unittest.skip("insertion doesn't work as expectedd")
    def test_insertion(self):
        for matrix in (getMatrixOfDoubles(True), getMatrixOfInts(True)):
            matrix.resize(10, 10)
            for initializer in range(7, 17):
                for index in range(10):
                    matrix.insertNonZeros(index, index, initializer)
                matrix.compress()
                diagonal = matrix.toSciPy().diagonal()
                expected_diagonal = np.array([initializer for _ in range(10)], dtype=diagonal.dtype)
                testing.assert_array_equal(diagonal, expected_diagonal)

    @unittest.skip("bugs in methods")
    def test_toSciPy(self):
        for matrix in (getMatrixOfDoubles(True), getMatrixOfInts(True)):
            matrix.resize(0, 0)
            with self.assertRaisesRegex(RuntimeError, "Empty matrices"):
                matrix.toSciPy()
            matrix.resize(5, 5)
            matrix.insertNonZeros(1, 1, 1)
            with self.assertRaisesRegex(RuntimeError, "Uncompressed matrices"):
                matrix.toSciPy()


if __name__ == '__main__':
    unittest.main()
