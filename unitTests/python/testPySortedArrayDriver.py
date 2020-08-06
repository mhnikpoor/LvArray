"""Tests for the SortedArray python wrapper."""

import unittest

import numpy as np

import testPythonSortedArray

class SortedArrayTests(unittest.TestCase):
    """Tests for the SortedArray python wrapper."""

    def test_insert(self):
        """Test the insert method."""
        return

    def test_remove(self):
        """Test the remove method."""
        return

    def test_modification_read_only(self):
        """Test that calling insert or remove on a read only SortedArray raises an exception."""
        return

    def test_modification_unsafe_conversion(self):
        """Test that calling insert or remove that involves an unsafe type conversion raises an exception."""
        return















y = testPythonSortedArray.getSortedArrayOfLongs( True )
y.insert( [0, 2, 4, 6] )
testPythonSortedArray.verifySortedArrayOfLongs( 4 )
print( y )
print( repr( y.toNumPy() ) )
print()

x = testPythonSortedArray.getSortedArrayOfInts( True )
x.insert( np.array( [2, 4, 6], np.int32 ) )
testPythonSortedArray.verifySortedArrayOfInts( 3 )
print( x )
print( repr( x.toNumPy() ) )
