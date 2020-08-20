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
#include "PyArray.hpp"
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
  PYTHON_ERROR_IF( self->array == nullptr, PyExc_RuntimeError, "The PyArray is not initialized.", nullptr )

#define VERIFY_MODIFIABLE( self ) \
  PYTHON_ERROR_IF( !self->array->modifiable(), PyExc_RuntimeError, "The PyArray is not modifiable.", nullptr )

struct PyArray
{
  PyObject_HEAD

  static constexpr char const * docString =
  "A Python interface to LvArray::Array.";

  internal::PyArrayWrapperBase * array;
};

/**
 * @brief The Python destructor for PyArray.
 * @param self The PyArray to destroy.
 */
static void PyArray_dealloc( PyArray * const self )
{ delete self->array; }

/**
 * @brief The Python __repr__ method for PyArray.
 * @param self The PyArray to represent.
 */
static PyObject * PyArray_repr( PyObject * const obj )
{
  PyArray * const self = convert< PyArray >( obj, getPyArrayType() );

  if ( self == nullptr )
  { return nullptr; }

  VERIFY_INITIALIZED( self );

  std::string const repr = self->array->repr();
  return PyUnicode_FromString( repr.c_str() );
}

static constexpr char const * PyArray_getSingleParameterResizeIndexDocString =
"getSingleParameterResizeIndex()\n"
"--\n\n"
"Return the default resize dimension.\n"
"\n"
"Returns\n"
"_______\n"
"int\n"
"    The default resize dimension.";
static PyObject * PyArray_getSingleParameterResizeIndex( PyArray * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( args );
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  return PyLong_FromLongLong( self->array->getSingleParameterResizeIndex() );
}

static constexpr char const * PyArray_setSingleParameterResizeIndexDocString =
"setSingleParameterResizeIndex(dim)\n"
"--\n\n"
"Set the default resize dimension.\n"
"\n"
"Returns\n"
"_______\n"
"None";
static PyObject * PyArray_setSingleParameterResizeIndex( PyArray * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  int dim;
  if ( !PyArg_ParseTuple( args, "i", &dim ) )
  { return nullptr; }

  PYTHON_ERROR_IF( dim < 0 || dim >= self->array->ndim(), PyExc_RuntimeError,
                   "Invalid argument to setSingleParameterResizeIndex.", nullptr );

  self->array->setSingleParameterResizeIndex( dim );

  Py_RETURN_NONE;
}

static constexpr char const * PyArray_resizeDocString =
"resize(newDim)\n"
"--\n\n"
"Resize the default dimension.\n"
"\n"
"Returns\n"
"_______\n"
"None";
static PyObject * PyArray_resize( PyArray * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  VERIFY_MODIFIABLE( self );

  long newSize;
  if ( !PyArg_ParseTuple( args, "l", &newSize ) )
  { return nullptr; }

  PYTHON_ERROR_IF( newSize < 0, PyExc_RuntimeError, "Size must be positive.", nullptr );

  self->array->resize( integerConversion< std::ptrdiff_t >( newSize ) );

  Py_RETURN_NONE;
}

static constexpr char const * PyArray_resizeAllDocString =
"resizeAll(newDims)\n"
"--\n\n"
"Resize all the dimensions.\n"
"\n"
"Returns\n"
"_______\n"
"None";
static PyObject * PyArray_resizeAll( PyArray * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  VERIFY_MODIFIABLE( self );

  PyObject * obj;
  if ( !PyArg_ParseTuple( args, "O", &obj ) )
  { return nullptr; }

  std::tuple< PyObjectRef< PyObject >, void const *, std::ptrdiff_t > ret =
    parseNumPyArray( obj, typeid( std::ptrdiff_t ) );

  if ( std::get< 0 >( ret ) == nullptr )
  { return nullptr; }


  PYTHON_ERROR_IF( std::get< 2 >( ret ) != self->array->ndim(), PyExc_RuntimeError,
                   "Invalid number of dimensions, should be " << self->array->ndim(), nullptr );

  self->array->resize( reinterpret_cast< std::ptrdiff_t const * >( std::get< 1 >( ret ) ) );

  Py_RETURN_NONE;
}

static constexpr char const * PyArray_toNumPyDocString =
"toNumPy()\n"
"--\n\n"
"Return a NumPy ndarray representing a shallow copy of the LvArray::Array.\n"
"\n"
"Returns\n"
"_______\n"
"NumPy ndarray\n"
"    A shallow copy of the LvArray::Array.";
static PyObject * PyArray_toNumPy( PyArray * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( args );

  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  return self->array->toNumPy();
}

BEGIN_ALLOW_DESIGNATED_INITIALIZERS

static PyMethodDef PyArray_methods[] = {
  { "getSingleParameterResizeIndex", (PyCFunction) PyArray_getSingleParameterResizeIndex, METH_NOARGS, PyArray_getSingleParameterResizeIndexDocString },
  { "setSingleParameterResizeIndex", (PyCFunction) PyArray_setSingleParameterResizeIndex, METH_VARARGS, PyArray_setSingleParameterResizeIndexDocString },
  { "resize", (PyCFunction) PyArray_resize, METH_VARARGS, PyArray_resizeDocString },
  { "resizeAll", (PyCFunction) PyArray_resizeAll, METH_VARARGS, PyArray_resizeAllDocString },
  { "toNumPy", (PyCFunction) PyArray_toNumPy, METH_VARARGS, PyArray_toNumPyDocString },
  { nullptr, nullptr, 0, nullptr } // Sentinel
};

static PyTypeObject PyArrayType = {
  PyVarObject_HEAD_INIT( nullptr, 0 )
  .tp_name = "LvArray.Array",
  .tp_basicsize = sizeof( PyArray ),
  .tp_itemsize = 0,
  .tp_dealloc = (destructor) PyArray_dealloc,
  .tp_repr = PyArray_repr,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyArray::docString,
  .tp_methods = PyArray_methods,
  .tp_new = PyType_GenericNew,
};

END_ALLOW_DESIGNATED_INITIALIZERS

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyTypeObject * getPyArrayType()
{ return &PyArrayType; }

namespace internal
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyObject * create( std::unique_ptr< PyArrayWrapperBase > && array )
{
  // Create a new Group and set the dataRepository::Group it points to.
  PyObject * const ret = PyObject_CallFunction( reinterpret_cast< PyObject * >( getPyArrayType() ), "" );
  PyArray * const retArray = reinterpret_cast< PyArray * >( ret );
  if ( retArray == nullptr )
  { return nullptr; }

  retArray->array = array.release();

  return ret;
}

} // namespace internal

} // namespace python
} // namespace LvArray
