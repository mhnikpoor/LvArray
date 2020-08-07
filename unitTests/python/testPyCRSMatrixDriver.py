"""Tests for the CRSMatrix python wrapper."""

import unittest

import numpy as np

import testPyCRSMatrix


matrix = testPyCRSMatrix.getMatrixOfDoubles( True )
matrix.resize( 10, 10 )

for i in range(10):
    matrix.insertNonZeros( i, i, 1 );

matrix.compress()

print( matrix.toSciPy().toarray() )
