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


// Python must be the first include.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Source includes
#include "PyArrayOfArrays.hpp"
#include "pythonHelpers.hpp"
#include "../limits.hpp"

// System includes
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#pragma GCC diagnostic pop

namespace LvArray
{
namespace python
{

#define VERIFY_NON_NULL_SELF( self ) \
  PYTHON_ERROR_IF( self == nullptr, PyExc_RuntimeError, "Passed a nullptr as self.", nullptr )

#define VERIFY_INITIALIZED( self ) \
  PYTHON_ERROR_IF( self->arrayOfArrays == nullptr, PyExc_RuntimeError, "The PyArrayOfArrays is not initialized.", nullptr )

#define VERIFY_MODIFIABLE( self ) \
  PYTHON_ERROR_IF( !self->arrayOfArrays->modifiable(), PyExc_RuntimeError, "The PyArrayOfArrays is not modifiable.", nullptr )

struct PyArrayOfArrays
{
  PyObject_HEAD

  static constexpr char const * docString =
  "Array of arrays";

  internal::PyArrayOfArraysWrapperBase * arrayOfArrays;
};

static void PyArrayOfArrays_dealloc( PyArrayOfArrays * const self )
{ delete self->arrayOfArrays; }

static PyObject * PyArrayOfArrays_repr( PyObject * const obj )
{
  PyArrayOfArrays * const self = convert< PyArrayOfArrays >( obj, getPyArrayOfArraysType() );


  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  std::string const repr = self->arrayOfArrays->repr();
  return PyUnicode_FromString( repr.c_str() );
}

static Py_ssize_t PyArrayOfArrays_len( PyArrayOfArrays * const self )
{
  PYTHON_ERROR_IF( self->arrayOfArrays == nullptr, PyExc_RuntimeError, "The PyArrayOfArrays is not initialized.", -1 );
  return integerConversion< Py_ssize_t >( self->arrayOfArrays->size() );
}

static PyObject * PyArrayOfArrays_getitem( PyArrayOfArrays * const self, PyObject * key )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  long long index = PyLong_AsLongLong( key );
  if ( PyErr_Occurred() != nullptr )
  { return nullptr; }
  if ( index < 0 || index >= self->arrayOfArrays->size() ){
    PyErr_SetString(PyExc_IndexError, "Index out of bounds");
    return nullptr;
  }
  return self->arrayOfArrays->operator[]( index );
}


BEGIN_ALLOW_DESIGNATED_INITIALIZERS

static PyMethodDef PyArrayOfArrays_methods[] = {
  { nullptr, nullptr, 0, nullptr } // Sentinel
};

static PyMappingMethods PyArrayOfArraysMappingMethods = {
  .mp_length = (lenfunc) PyArrayOfArrays_len,
  .mp_subscript = (binaryfunc) PyArrayOfArrays_getitem,
};

static PyTypeObject PyArrayOfArraysType = {
  PyVarObject_HEAD_INIT( nullptr, 0 )
  .tp_name = "ArrayOfArrays",
  .tp_basicsize = sizeof( PyArrayOfArrays ),
  .tp_itemsize = 0,
  .tp_dealloc = (destructor) PyArrayOfArrays_dealloc,
  .tp_repr = PyArrayOfArrays_repr,
  .tp_as_mapping = &PyArrayOfArraysMappingMethods,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyArrayOfArrays::docString,
  .tp_methods = PyArrayOfArrays_methods,
  .tp_new = PyType_GenericNew,
};

END_ALLOW_DESIGNATED_INITIALIZERS

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyTypeObject * getPyArrayOfArraysType()
{ return &PyArrayOfArraysType; }

namespace internal
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyObject * create( std::unique_ptr< PyArrayOfArraysWrapperBase > && array )
{
  // Create a new Group and set the dataRepository::Group it points to.
  PyObject * const ret = PyObject_CallFunction( reinterpret_cast< PyObject * >( getPyArrayOfArraysType() ), "" );
  PyArrayOfArrays * const retArrayOfArrays = reinterpret_cast< PyArrayOfArrays * >( ret );
  if ( retArrayOfArrays == nullptr )
  { return nullptr; }

  retArrayOfArrays->arrayOfArrays = array.release();

  return ret;
}

} // namespace internal

} // namespace python
} // namespace LvArray
