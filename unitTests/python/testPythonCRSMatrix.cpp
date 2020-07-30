/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2019, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL LvArray_ARRAY_API

#include "MallocBuffer.hpp"
#include "CRSMatrix.hpp"
#include "python/numpyCRSMatrixView.hpp"

#include <Python.h>
#include <numpy/arrayobject.h>

#define NUMROWS 10
#define NUMCOLS 10

// global SortedArray of ints
static LvArray::CRSMatrix< double, int, std::ptrdiff_t, LvArray::MallocBuffer > crsMat(NUMROWS, NUMCOLS, 10);

static LvArray::CRSMatrix< double, int, std::ptrdiff_t, LvArray::MallocBuffer > uncompressedMat(NUMROWS, NUMCOLS, 10);

static LvArray::CRSMatrix< double, int, std::ptrdiff_t, LvArray::MallocBuffer > dim0Mat(0, 0, 10);

/**
 * Return a 3-tuple of numpy arrays representing the CRSMatrix.
 */
static PyObject * getMatrix( PyObject * self, PyObject * args ){
    LVARRAY_UNUSED_VARIABLE( self );
    LVARRAY_UNUSED_VARIABLE( args );
    return LvArray::python::create( crsMat.toViewConstSizes(), true );
}

/**
 * Initialize the global CRSMatrix along the diagonal
 */
static PyObject * initMatrix( PyObject * self, PyObject * args ){
    LVARRAY_UNUSED_VARIABLE( self );
    double initializer;
    if ( !PyArg_ParseTuple( args, "d", &initializer ) )
        return NULL;
    for ( std::ptrdiff_t row = 0; row < crsMat.numRows(); ++row )
    {
        crsMat.removeNonZero( row, row );
        crsMat.insertNonZero( row, row, initializer );
    }
    crsMat.compress();
    return getMatrix( nullptr, nullptr );
}

/**
 * Initialize the global uncompressed CRSMatrix along the diagonal
 */
static PyObject * initUncompressedMatrix( PyObject * self, PyObject * args ){
    LVARRAY_UNUSED_VARIABLE( self );
    double initializer;
    if ( !PyArg_ParseTuple( args, "d", &initializer ) )
        return NULL;
    for ( std::ptrdiff_t row = 0; row < uncompressedMat.numRows(); ++row )
    {
        uncompressedMat.removeNonZero( row, row );
        uncompressedMat.insertNonZero( row, row, initializer );
    }
    return LvArray::python::create( uncompressedMat.toViewConstSizes(), true );
}

/**
 * Return a 3-tuple of numpy arrays representing the CRSMatrix.
 */
static PyObject * get0DimMatrix( PyObject * self, PyObject * args ){
    LVARRAY_UNUSED_VARIABLE( self );
    LVARRAY_UNUSED_VARIABLE( args );
    return LvArray::python::create( dim0Mat.toViewConstSizes(), true );
}

/**
 * Array of functions and docstrings to export to Python
 */
static PyMethodDef CRSMatrixFuncs[] = {
    {"get_matrix",  getMatrix, METH_NOARGS,
     "Return the numpy representation of a global CRSMatrix."},
    {"init_matrix",  initMatrix, METH_VARARGS,
     "Initialize a global CRSMatrix along the diagonal with the given float."},
    {"init_uncompressed",  initUncompressedMatrix, METH_VARARGS,
     "Initialize a global CRSMatrix along the diagonal with the given float."},
    {"get_dim0",  get0DimMatrix, METH_VARARGS,
     "Initialize a global CRSMatrix along the diagonal with the given float."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/**
 * Initialize the module object for Python with the exported functions
 */
static struct PyModuleDef testPythonCRSMatrixModule = {
    PyModuleDef_HEAD_INIT,
    "testPythonArray",   /* name of module */
    /* module documentation, may be NULL */
    "Module for testing numpy views of LvArray::CRSMatrix objects",
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    CRSMatrixFuncs,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC
PyInit_testPythonCRSMatrix(void)
{
    import_array();
    PyObject * module = PyModule_Create(&testPythonCRSMatrixModule);
    PyModule_AddIntMacro( module, NUMROWS );
    PyModule_AddIntMacro( module, NUMCOLS );
    return module;
}
