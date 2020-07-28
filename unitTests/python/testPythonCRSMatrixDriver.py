"""Tests for the `lvarray` extension module"""

import unittest

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csr
from numpy import testing

import testPythonCRSMatrix as lvarray


class LvArrayCRSMatrix(csr.csr_matrix):

    def __setitem__(self, key, value):
        try:
            row, col = key
        except (TypeError, ValueError):
            self.data[self.indptr[key] : self.indptr[key + 1]] = value
        else:
            cols_in_row = self.indices[self.indptr[row] : self.indptr[row + 1]]
            candidate_index = np.searchsorted(cols_in_row, col)
            if candidate_index == len(cols_in_row) or cols_in_row[candidate_index] != col:
                raise IndexError(
                    f"Entry {key} is not represented in this {type(self).__name__} "
                    "and adding new values is not supported"
                )
            self.data[self.indptr[row] + candidate_index] = value


class ArrayTests(unittest.TestCase):
    """Tests for the `lvarray` extension module"""

    def test_init(self):
        """Test that the array is properly initialized along the diagonal"""
        for initializer in range(-20, -1):
            mat = LvArrayCRSMatrix(lvarray.init_matrix(initializer))
            testing.assert_array_equal(
                mat.diagonal(),
                np.array([float(initializer) for _ in range(lvarray.NUMROWS)]),
            )

    def test_modify(self):
        for initializer in range(5, 15):
            mat = LvArrayCRSMatrix(lvarray.init_matrix(initializer))
            for row in range(lvarray.NUMROWS):
                for col in range(lvarray.NUMCOLS):
                    if row == col:
                        mat[row, col] = 17.0
                        self.assertEqual(mat[row, col], 17.0)
                        testing.assert_array_equal(
                            mat.toarray(),
                            LvArrayCRSMatrix(lvarray.get_matrix()).toarray(),
                        )
                    else:
                        with self.assertRaisesRegex(
                            IndexError, "adding new values is not supported"
                        ):
                            mat[row, col] = 17.0

    def test_get_dim0(self):
        with self.assertRaisesRegex(RuntimeError, "0-dimensional matrices"):
            lvarray.get_dim0()

    def test_init_uncompressed(self):
        for initializer in range(-20, -1):
            with self.assertRaisesRegex(RuntimeError, "Uncompressed matrices"):
                LvArrayCRSMatrix(lvarray.init_uncompressed(initializer))


if __name__ == "__main__":
    unittest.main()
