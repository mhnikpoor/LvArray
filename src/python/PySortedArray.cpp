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
#include "PySortedArray.hpp"
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
  PYTHON_ERROR_IF( self->sortedArray == nullptr, PyExc_RuntimeError, "The PySortedArray is not initialized.", nullptr )

#define VERIFY_MODIFIABLE( self ) \
  PYTHON_ERROR_IF( !self->sortedArray->modifiable(), PyExc_RuntimeError, "The PySortedArray is not modifiable.", nullptr )

struct PySortedArray
{
  PyObject_HEAD

  static constexpr char const * docString =
  "";

  internal::PySortedArrayWrapperBase * sortedArray;
};


static void PySortedArray_dealloc( PySortedArray * const self )
{ delete self->sortedArray; }


static PyObject * PySortedArray_repr( PyObject * const obj )
{
  PySortedArray * const self = convert< PySortedArray >( obj, getPySortedArrayType() );

  if ( self == nullptr )
  { return nullptr; }

  VERIFY_INITIALIZED( self );

  std::string const repr = self->sortedArray->repr();
  return PyUnicode_FromString( repr.c_str() );
}

static constexpr char const * PySortedArray_insertDocString =
"";
static PyObject * PySortedArray_insert( PySortedArray * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  VERIFY_MODIFIABLE( self );

  std::tuple< PyObjectRef< PyObject >, void const *, std::ptrdiff_t > ret =
    parseNumPyArray( args, self->sortedArray->valueType() );

  if ( std::get< 0 >( ret ) == nullptr )
  { return nullptr; }

  return PyLong_FromLongLong( self->sortedArray->insert( std::get< 1 >( ret ), std::get< 2 >( ret ) ) );
}

static constexpr char const * PySortedArray_removeDocString =
"";
static PyObject * PySortedArray_remove( PySortedArray * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  VERIFY_MODIFIABLE( self );

  std::tuple< PyObjectRef< PyObject >, void const *, std::ptrdiff_t > ret =
    parseNumPyArray( args, self->sortedArray->valueType() );

  if ( std::get< 0 >( ret ) == nullptr )
  { return nullptr; }

  return PyLong_FromLongLong( self->sortedArray->remove( std::get< 1 >( ret ), std::get< 2 >( ret ) ) );
}


static constexpr char const * PySortedArray_toNumPyDocString =
"";
static PyObject * PySortedArray_toNumPy( PySortedArray * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( args );

  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  return self->sortedArray->toNumPy();
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


static PyMethodDef PySortedArray_methods[] = {
  { "insert", (PyCFunction) PySortedArray_insert, METH_VARARGS, PySortedArray_insertDocString },
  { "remove", (PyCFunction) PySortedArray_remove, METH_VARARGS, PySortedArray_removeDocString },
  { "toNumPy", (PyCFunction) PySortedArray_toNumPy, METH_VARARGS, PySortedArray_toNumPyDocString },
  { nullptr, nullptr, 0, nullptr } // Sentinel
};

static PyTypeObject PySortedArrayType = {
  PyVarObject_HEAD_INIT( nullptr, 0 )
  .tp_name = "SortedArray",
  .tp_basicsize = sizeof( PySortedArray ),
  .tp_itemsize = 0,
  .tp_dealloc = (destructor) PySortedArray_dealloc,
  .tp_repr = PySortedArray_repr,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PySortedArray::docString,
  .tp_methods = PySortedArray_methods,
  .tp_new = PyType_GenericNew,
};

#pragma GCC diagnostic pop

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyTypeObject * getPySortedArrayType()
{ return &PySortedArrayType; }

namespace internal
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyObject * create( std::unique_ptr< PySortedArrayWrapperBase > && array )
{
  // Create a new Group and set the dataRepository::Group it points to.
  PyObject * const ret = PyObject_CallFunction( reinterpret_cast< PyObject * >( getPySortedArrayType() ), "" );
  PySortedArray * const retSortedArray = reinterpret_cast< PySortedArray * >( ret );
  if ( retSortedArray == nullptr )
  { return nullptr; }

  retSortedArray->sortedArray = array.release();

  return ret;
}

} // namespace internal

} // namespace python
} // namespace LvArray
