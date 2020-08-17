import unittest

import numpy as np

from testPyArrayOfArrays import get_array_of_arrays

class LvArrayArrayOfArraysTests(unittest.TestCase):

    def test_modify(self):
        arr = get_array_of_arrays(True)
        breakpoint()
        print(arr)
        x = len(arr)
        print(x)



if __name__ == '__main__':
    unittest.main()
