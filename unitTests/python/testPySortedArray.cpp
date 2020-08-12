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

#include "python/PySortedArray.hpp"
#include "MallocBuffer.hpp"

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

BEGIN_ALLOW_DESIGNATED_INITIALIZERS

/**
 * Array of functions and docstrings to export to Python
 */
static PyMethodDef testPySortedArrayFuncs[] = {
  {"get_sorted_array_int", getSortedArrayOfInts, METH_VARARGS, ""},
  {"get_sorted_array_long", getSortedArrayOfLongs, METH_VARARGS, ""},
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

END_ALLOW_DESIGNATED_INITIALIZERS

PyMODINIT_FUNC
PyInit_testPySortedArray(void)
{
  LvArray::python::PyObjectRef<> module = PyModule_Create( &testPySortedArrayModule );

  if ( !LvArray::python::addTypeToModule( module, LvArray::python::getPySortedArrayType(), "SortedArray" ) )
  { return nullptr; }

  return module.release();
}
