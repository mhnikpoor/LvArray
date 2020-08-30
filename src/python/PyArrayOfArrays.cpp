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

static PyObject * PyArrayOfArrays_getitem( PyArrayOfArrays * const self, Py_ssize_t pyindex )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  long long index = integerConversion< long long >( pyindex );
  if ( PyErr_Occurred() != nullptr )
  { return nullptr; }
  if ( index < 0 || index >= self->arrayOfArrays->size() ){
    PyErr_SetString(PyExc_IndexError, "Index out of bounds");
    return nullptr;
  }
  return self->arrayOfArrays->operator[]( index );
}

static int PyArrayOfArrays_delitem( PyArrayOfArrays * const self, Py_ssize_t pyindex, PyObject * val ){
  if ( val != nullptr ){
    PyErr_SetString(PyExc_TypeError, "ArrayOfArrays object does not support item assignment");
    return -1;
  }
  long long index = integerConversion< long long >( pyindex );
  if ( PyErr_Occurred() != nullptr )
  { return -1; }
  if ( index < 0 || index >= self->arrayOfArrays->size() ){
    PyErr_SetString(PyExc_IndexError, "Index out of bounds");
    return -1;
  }
  self->arrayOfArrays->eraseArray( index );
  return 0;
}

static PyObject * PyArrayOfArrays_sq_concat( PyArrayOfArrays * self, PyObject * args ){
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  Py_RETURN_NOTIMPLEMENTED;
}

static PyObject * PyArrayOfArrays_sq_repeat( PyArrayOfArrays * self, Py_ssize_t args ){
  LVARRAY_UNUSED_VARIABLE( self );
  LVARRAY_UNUSED_VARIABLE( args );
  Py_RETURN_NOTIMPLEMENTED;
}

static constexpr char const * PyArrayOfArrays_insertIntoDocString =
"";
static PyObject * PyArrayOfArrays_insertIntoArray( PyArrayOfArrays * self, PyObject * args ){
  long long array, index;
  PyObject * arr;
  if( !PyArg_ParseTuple( args, "LLO", &array, &index, &arr ) )
  { return nullptr; }
  if ( array < 0 || array >= self->arrayOfArrays->size() || index < 0
       || index > self->arrayOfArrays->sizeOfArray( array )){
    PyErr_SetString(PyExc_IndexError, "index out of bounds");
    return nullptr;
  }
  std::tuple< PyObjectRef< PyObject >, void const *, std::ptrdiff_t > ret =
    parseNumPyArray( arr, self->arrayOfArrays->valueType() );
  if ( std::get< 0 >( ret ) == nullptr )
  { return nullptr; }
  self->arrayOfArrays->insertIntoArray( array, index, std::get< 1 >( ret ), std::get< 2 >( ret ) );
  Py_RETURN_NONE;
}

static constexpr char const * PyArrayOfArrays_insertDocString =
"";
static PyObject * PyArrayOfArrays_insert( PyArrayOfArrays * self, PyObject * args ){
  long long index;
  PyObject * arr;
  if( !PyArg_ParseTuple( args, "LO", &index, &arr ) )
  { return nullptr; }
  if ( index < 0 || index > self->arrayOfArrays->size() ){
    PyErr_SetString(PyExc_IndexError, "Index out of bounds");
    return nullptr;
  }
  std::tuple< PyObjectRef< PyObject >, void const *, std::ptrdiff_t > ret =
    parseNumPyArray( arr, self->arrayOfArrays->valueType() );
  if ( std::get< 0 >( ret ) == nullptr )
  { return nullptr; }
  self->arrayOfArrays->insertArray( index, std::get< 1 >( ret ), std::get< 2 >( ret ) );
  Py_RETURN_NONE;
}

static constexpr char const * PyArrayOfArrays_eraseFromDocString =
"";
static PyObject * PyArrayOfArrays_eraseFrom( PyArrayOfArrays * self, PyObject * args ){
  long long index, begin;
  if( !PyArg_ParseTuple( args, "LL", &index, &begin ) )
  { return nullptr; }
  if ( index < 0 || index >= self->arrayOfArrays->size() || begin < 0
       || begin >= self->arrayOfArrays->sizeOfArray( index ) ){
    PyErr_SetString(PyExc_IndexError, "index out of bounds");
    return nullptr;
  }
  self->arrayOfArrays->eraseFromArray( index, begin );
  Py_RETURN_NONE;
}


BEGIN_ALLOW_DESIGNATED_INITIALIZERS

static PyMethodDef PyArrayOfArrays_methods[] = {
  { "insert", (PyCFunction) PyArrayOfArrays_insert, METH_VARARGS, PyArrayOfArrays_insertDocString },
  { "insert_into", (PyCFunction) PyArrayOfArrays_insertIntoArray, METH_VARARGS, PyArrayOfArrays_insertIntoDocString },
  { "erase_from", (PyCFunction) PyArrayOfArrays_eraseFrom, METH_VARARGS, PyArrayOfArrays_eraseFromDocString },
  { nullptr, nullptr, 0, nullptr } // Sentinel
};

static PySequenceMethods PyArrayOfArraysSequenceMethods = {
  .sq_length = (lenfunc) PyArrayOfArrays_len,
  .sq_concat = (binaryfunc) PyArrayOfArrays_sq_concat,
  .sq_repeat = (ssizeargfunc) PyArrayOfArrays_sq_repeat,
  .sq_item = (ssizeargfunc) PyArrayOfArrays_getitem,
  .sq_ass_item = (ssizeobjargproc) PyArrayOfArrays_delitem,
};

static PyTypeObject PyArrayOfArraysType = {
  PyVarObject_HEAD_INIT( nullptr, 0 )
  .tp_name = "pylvarray.ArrayOfArrays",
  .tp_basicsize = sizeof( PyArrayOfArrays ),
  .tp_itemsize = 0,
  .tp_dealloc = (destructor) PyArrayOfArrays_dealloc,
  .tp_repr = PyArrayOfArrays_repr,
  .tp_as_sequence = &PyArrayOfArraysSequenceMethods,
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
