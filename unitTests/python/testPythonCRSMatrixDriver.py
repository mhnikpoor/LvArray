"""Tests for the `lvarray` extension module"""

import unittest

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csr
from numpy import testing

import testPythonCRSMatrix as lvarray


class LvArrayCRSMatrix(csr.csr_matrix):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__valid_indices = set()
        for row_index in range(self.shape[0]):
            col_indices = self.indices[self.indptr[row_index] : self.indptr[row_index + 1]]
            for col_index in col_indices:
                self.__valid_indices.add((row_index, col_index))

    def __setitem__(self, key, value):
        super().__getitem__(key) # check for IndexError
        if key not in self.__valid_indices:
            raise IndexError(
                f"Entry {key} is not represented in this {type(self).__name__} "
                "and adding new values is not supported"
            )
        return super().__setitem__(key, value)


class ArrayTests(unittest.TestCase):
    """Tests for the `lvarray` extension module"""

    def test_init(self):
        """Test that the array is properly initialized along the diagonal"""
        for initializer in range(-20, -1):
            mat = LvArrayCRSMatrix(lvarray.init_matrix(initializer))
            testing.assert_array_equal(mat.diagonal(), np.array([float(initializer) for _ in range(lvarray.NUMROWS)]))

    def test_modify(self):
        for initializer in range(5, 15):
            mat = LvArrayCRSMatrix(lvarray.init_matrix(initializer))
            for row in range(lvarray.NUMROWS):
                for col in range(lvarray.NUMCOLS):
                    if row == col:
                        mat[row, col] = 17.0
                        self.assertEqual(mat[row, col], 17.0)
                    else:
                        with self.assertRaisesRegex(IndexError, "adding new values is not supported"):
                            mat[row, col] = 17.0



if __name__ == "__main__":
    unittest.main()
