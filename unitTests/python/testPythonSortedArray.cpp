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
#define NPY_NO_DEPRECATED_API NPY_1_15_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL LvArray_ARRAY_API

#include "SortedArray.hpp"
#include "MallocBuffer.hpp"
#include "python/numpySortedArrayView.hpp"

#include <Python.h>
#include <numpy/arrayobject.h>

// global SortedArray of ints
static LvArray::SortedArray< int, std::ptrdiff_t, LvArray::MallocBuffer > sorted_arr_int;

/**
 * Fetch the global SortedArray and return a numpy view of it.
 */
static PyObject *
get_sorted_array_int( PyObject *self, PyObject *args )
{
    LVARRAY_UNUSED_VARIABLE( self );
    LVARRAY_UNUSED_VARIABLE( args );
    return LvArray::python::create( sorted_arr_int.toView(), true );
}

/**
 * Clear and initialize the global SortedArray with `range(start, stop)`
 * NOTE: this may invalidate previous views of the array!
 */
static PyObject *
set_sorted_array_int( PyObject *self, PyObject *args )
{
    LVARRAY_UNUSED_VARIABLE( self );
    int start;
    int stop;
    if ( !PyArg_ParseTuple( args, "ii", &start, &stop ) )
        return NULL;
    if ( stop <= start ){
        Py_RETURN_NONE;
    }
    sorted_arr_int.clear();
    for ( int i = start; i < stop; ++i )
    {
        sorted_arr_int.insert( i );
    }
    return get_sorted_array_int( nullptr, nullptr );
}

/**
 * Multiply the global SortedArray by a factor. Return None.
 */
static PyObject *
multiply_sorted_array_int( PyObject *self, PyObject *args )
{
    LVARRAY_UNUSED_VARIABLE( self );
    int factor;
    if ( !PyArg_ParseTuple( args, "i", &factor ) )
        return NULL;
    LvArray::SortedArray< int, std::ptrdiff_t, LvArray::MallocBuffer > newSortedArray;
    for (int i : sorted_arr_int)
    {
        newSortedArray.insert( factor * i );
    }
    sorted_arr_int = newSortedArray;
    Py_RETURN_NONE;
}

/**
 * Array of functions and docstrings to export to Python
 */
static PyMethodDef SortedArrayFuncs[] = {
    {"set_sorted_array_int",  set_sorted_array_int, METH_VARARGS,
     "Return the numpy representation of a SortedArray initialized like `range(start, stop)`."},
    {"get_sorted_array_int",  get_sorted_array_int, METH_NOARGS,
     "Get the numpy representation of a SortedArray."},
    {"multiply_sorted_array_int",  multiply_sorted_array_int, METH_VARARGS,
     "Multiply the contents of a SortedArray."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/**
 * Initialize the module object for Python with the exported functions
 */
static struct PyModuleDef testPythonSortedArraymodule = {
    PyModuleDef_HEAD_INIT,
    "testPythonArray",   /* name of module */
    /* module documentation, may be NULL */
    "Module for testing numpy views of LvArray::SortedArray objects",
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SortedArrayFuncs,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC
PyInit_testPythonSortedArray(void)
{
    import_array();
    return PyModule_Create(&testPythonSortedArraymodule);
}
