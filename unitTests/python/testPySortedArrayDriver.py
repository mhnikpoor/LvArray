"""Tests for the SortedArray python wrapper."""

import unittest

import numpy as np
from numpy import testing

from testPySortedArray import get_sorted_array_int, get_sorted_array_long


def clear(arr):
    """Test the insert method."""
    dtype = arr.to_numpy().dtype.type
    while len(arr.to_numpy()) > 0:
        arr.remove(arr.to_numpy()[0])


class SortedArrayTests(unittest.TestCase):
    """Tests for the SortedArray python wrapper."""

    lvarrays = (get_sorted_array_int, get_sorted_array_long)

    def setUp(self):
        """Test the insert method."""
        for getter in self.lvarrays:
            clear(getter(True))
            self.assertEqual(len(getter(True).to_numpy()), 0)

    def test_insert(self):
        for getter in self.lvarrays:
            arr = getter(True)
            dtype = arr.to_numpy().dtype.type
            for value in range(-5, 15):
                arr.insert(dtype(value))
                self.assertIn(dtype(value), arr.to_numpy())
            self.assertEqual(len(arr.to_numpy()), 20)
            testing.assert_array_equal(arr.to_numpy(), np.sort(arr.to_numpy()))

    def test_remove(self):
        """Test the remove method."""
        for getter in self.lvarrays:
            arr = getter(True)
            dtype = arr.to_numpy().dtype.type
            for value in range(-5, 15):
                arr.insert(dtype(value))
            self.assertEqual(len(arr.to_numpy()), 20)
            for value in range(-5, 15):
                arr.remove(dtype(value))
            self.assertEqual(len(arr.to_numpy()), 0)

    def test_modification_read_only(self):
        """Test that calling insert or remove on a read only SortedArray raises an exception."""
        for getter in self.lvarrays:
            arr = getter(True)
            dtype = arr.to_numpy().dtype.type
            arr.insert(dtype(5))
            self.assertEqual(arr.to_numpy()[0], dtype(5))
            with self.assertRaisesRegex(ValueError, "read-only"):
                arr.to_numpy()[0] = dtype(6)

    def test_modification_unsafe_conversion(self):
        """Test that calling insert or remove that involves an unsafe type conversion raises an exception."""
        for getter in self.lvarrays:
            arr = getter(True)
            with self.assertRaisesRegex(TypeError, "Cannot safely convert"):
                arr.insert(5.6)


if __name__ == '__main__':
    unittest.main()
