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
#include <Python.h>

#include "SortedArray.hpp"
#include "MallocBuffer.hpp"
#include "python/PySortedArray.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// global SortedArray of ints
static LvArray::SortedArray< int, std::ptrdiff_t, LvArray::MallocBuffer > sortedArrayOfInts;
static LvArray::SortedArray< long, std::ptrdiff_t, LvArray::MallocBuffer > sortedArrayOfLongs;

static PyObject * getSortedArrayOfInts( PyObject * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  
  int modify;
  if( !PyArg_ParseTuple( args, "p", &modify ) )
  { return nullptr; }

  return LvArray::python::create( sortedArrayOfInts, modify );
}

static PyObject * getSortedArrayOfLongs( PyObject * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( self );
  
  int modify;
  if( !PyArg_ParseTuple( args, "p", &modify ) )
  { return nullptr; }

  return LvArray::python::create( sortedArrayOfLongs, modify );
}

// Allow mixing designated and non-designated initializers in the same initializer list.
// I don't like the pragmas but the designated initializers is the only sane way to do this stuff.
// The other option is to put this in a `.c` file and compile with the C compiler, but that seems like more work.
#pragma GCC diagnostic push
#if defined( __clang_version__ )
  #pragma GCC diagnostic ignored "-Wc99-designator"
#else
  #pragma GCC diagnostic ignored "-Wpedantic"
  #pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

/**
 * Array of functions and docstrings to export to Python
 */
static PyMethodDef testPySortedArrayFuncs[] = {
  {"getSortedArrayOfInts", getSortedArrayOfInts, METH_VARARGS, ""},
  {"getSortedArrayOfLongs", getSortedArrayOfLongs, METH_VARARGS, ""},
  {nullptr, nullptr, 0, nullptr}        /* Sentinel */
};

/**
 * Initialize the module object for Python with the exported functions
 */
static struct PyModuleDef testPySortedArrayModule = {
  PyModuleDef_HEAD_INIT,
  .m_name = "testPySortedArray",
  .m_doc = "",
  .m_size = -1,
  .m_methods = testPySortedArrayFuncs,
};

#pragma GCC diagnostic pop

PyMODINIT_FUNC
PyInit_testPySortedArray(void)
{
  import_array();

  if( PyType_Ready( LvArray::python::getPySortedArrayType() ) < 0 )
  { return nullptr; }

  PyObject * module = PyModule_Create( &testPySortedArrayModule );

  Py_INCREF( LvArray::python::getPySortedArrayType() );
  if ( PyModule_AddObject( module, "SortedArray", reinterpret_cast< PyObject * >( LvArray::python::getPySortedArrayType() ) ) < 0 )
  {
    Py_DECREF( LvArray::python::getPySortedArrayType() );
    Py_DECREF( module );
    return nullptr;
  }

  return module;
}
