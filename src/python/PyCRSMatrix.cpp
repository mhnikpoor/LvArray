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
#include "PyCRSMatrix.hpp"
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
  PYTHON_ERROR_IF( self->matrix == nullptr, PyExc_RuntimeError, "The PyCRSMatrix is not initialized.", nullptr )

#define VERIFY_MODIFIABLE( self ) \
  PYTHON_ERROR_IF( !self->matrix->modifiable(), PyExc_RuntimeError, "The PyCRSMatrix is not modifiable.", nullptr )

struct PyCRSMatrix
{
  PyObject_HEAD

  static constexpr char const * docString =
  "";

  internal::PyCRSMatrixWrapperBase * matrix;
};


static void PyCRSMatrix_dealloc( PyCRSMatrix * const self )
{ delete self->matrix; }


static PyObject * PyCRSMatrix_repr( PyObject * const obj )
{
  PyCRSMatrix * const self = convert< PyCRSMatrix >( obj, getPyCRSMatrixType() );

  if ( self == nullptr )
  { return nullptr; }

  VERIFY_INITIALIZED( self );

  std::string const repr = self->matrix->repr();
  return PyUnicode_FromString( repr.c_str() );
}

static constexpr char const * PyCRSMatrix_numRowsDocString =
"";
static PyObject * PyCRSMatrix_numRows( PyCRSMatrix * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( args );
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  return PyLong_FromLongLong( self->matrix->numRows() );
}

static constexpr char const * PyCRSMatrix_numColumnsDocString =
"";
static PyObject * PyCRSMatrix_numColumns( PyCRSMatrix * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( args );
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  return PyLong_FromLongLong( self->matrix->numColumns() );
}

static constexpr char const * PyCRSMatrix_getColumnsDocString =
"";
static PyObject * PyCRSMatrix_getColumns( PyCRSMatrix * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  long row;
  if ( !PyArg_ParseTuple( args, "l", &row ) )
  { return nullptr; }

  std::ptrdiff_t const numRows = self->matrix->numRows();
  PYTHON_ERROR_IF( row < 0 || row >= numRows, PyExc_RuntimeError,
                   "Row " << row  <<" is out of bounds for a matrix with " << numRows << " rows.", nullptr );

  return self->matrix->getColumns( row );
}

static constexpr char const * PyCRSMatrix_getEntriesDocString =
"";
static PyObject * PyCRSMatrix_getEntries( PyCRSMatrix * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  long row;
  if ( !PyArg_ParseTuple( args, "l", &row ) )
  { return nullptr; }

  std::ptrdiff_t const numRows = self->matrix->numRows();
  PYTHON_ERROR_IF( row < 0 || row >= numRows, PyExc_RuntimeError,
                   "Row " << row << " is out of bounds for a matrix with " << numRows << " rows.", nullptr );

  return self->matrix->getEntries( row );
}

static constexpr char const * PyCRSMatrix_toSciPyDocString =
"";
static PyObject * PyCRSMatrix_toSciPy( PyCRSMatrix * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( args );
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );

  PYTHON_ERROR_IF( !self->matrix->isCompressed(), PyExc_RuntimeError,
                   "Uncompressed matrices can not be exported to SciPy", nullptr );

  PYTHON_ERROR_IF( self->matrix->numRows() == 0, PyExc_RuntimeError,
                   "Empty matrices cannot be exported to SciPy", nullptr );

  std::array< PyObject *, 3 > const triple = self->matrix->getEntriesColumnsAndOffsets();

  PYTHON_ERROR_IF( triple[ 0 ] == nullptr, PyExc_RuntimeError,
                   "Error constructing the entries NumPy array", nullptr );

  PYTHON_ERROR_IF( triple[ 1 ] == nullptr, PyExc_RuntimeError,
                   "Error constructing the columns NumPy array", nullptr );

  PYTHON_ERROR_IF( triple[ 2 ] == nullptr, PyExc_RuntimeError,
                   "Error constructing the offsets NumPy array", nullptr );

  PyObjectRef<> sciPySparse = PyImport_ImportModule( "scipy.sparse" );
  PyObjectRef<> constructor = PyObject_GetAttrString( sciPySparse, "csr_matrix" );

  return PyObject_CallFunction( constructor,
                                "(OOO)(ll)",
                                triple[ 0 ],
                                triple[ 1 ],
                                triple[ 2 ],
                                self->matrix->numRows(),
                                self->matrix->numColumns() );
}

static constexpr char const * PyCRSMatrix_resizeDocString =
"";
static PyObject * PyCRSMatrix_resize( PyCRSMatrix * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  VERIFY_MODIFIABLE( self );

  long numRows, numCols, initialRowCapacity=0;
  if ( !PyArg_ParseTuple( args, "ll|l", &numRows, &numCols, &initialRowCapacity ) )
  { return nullptr; }

  PYTHON_ERROR_IF( numRows < 0 || numCols < 0 || initialRowCapacity < 0, PyExc_RuntimeError,
                   "Arguments must be positive.", nullptr );

  self->matrix->resize( numRows, numCols, initialRowCapacity );

  Py_RETURN_NONE;
}

static constexpr char const * PyCRSMatrix_compressDocString =
"";
static PyObject * PyCRSMatrix_compress( PyCRSMatrix * const self, PyObject * const args )
{
  LVARRAY_UNUSED_VARIABLE( args );
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  VERIFY_MODIFIABLE( self );

  self->matrix->compress();

  Py_RETURN_NONE;
}

static constexpr char const * PyCRSMatrix_insertNonZerosDocString =
"";
static PyObject * PyCRSMatrix_insertNonZeros( PyCRSMatrix * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  VERIFY_MODIFIABLE( self );

  long row;
  PyObject * cols, * entries;
  if ( !PyArg_ParseTuple( args, "lOO", &row, &cols, &entries ) )
  { return nullptr; }

  std::tuple< PyObjectRef< PyObject >, void const *, std::ptrdiff_t > colsInfo =
    parseNumPyArray( cols, self->matrix->columnType() );

  std::tuple< PyObjectRef< PyObject >, void const *, std::ptrdiff_t > entriesInfo =
    parseNumPyArray( entries, self->matrix->entryType() );

  if ( std::get< 0 >( colsInfo ) == nullptr ||  std::get< 0 >( entriesInfo ) == nullptr )
  { return nullptr; }

  PYTHON_ERROR_IF( std::get< 2 >( colsInfo ) != std::get< 2 >( entriesInfo ), PyExc_RuntimeError,
                   "Number of columns and entries must match!", nullptr );

  std::ptrdiff_t const numRows = self->matrix->numRows();
  PYTHON_ERROR_IF( row < 0 || row >= numRows, PyExc_RuntimeError,
                   "Row " << row << " is out of bounds for a matrix with " << numRows << " rows.", nullptr );

  return PyLong_FromLongLong( self->matrix->insertNonZeros( row,
                                                            std::get< 1 >( colsInfo ),
                                                            std::get< 1 >( entriesInfo ),
                                                            std::get< 2 >( entriesInfo ) ) );
}

static constexpr char const * PyCRSMatrix_removeNonZerosDocString =
"";
static PyObject * PyCRSMatrix_removeNonZeros( PyCRSMatrix * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  VERIFY_MODIFIABLE( self );

  long row;
  PyObject * cols;
  if ( !PyArg_ParseTuple( args, "lO", &row, &cols ) )
  { return nullptr; }

  std::tuple< PyObjectRef< PyObject >, void const *, std::ptrdiff_t > colsInfo =
    parseNumPyArray( cols, self->matrix->columnType() );

  if ( std::get< 0 >( colsInfo ) == nullptr )
  { return nullptr; }

  std::ptrdiff_t const numRows = self->matrix->numRows();
  PYTHON_ERROR_IF( row < 0 || row >= numRows, PyExc_RuntimeError,
                   "Row " << row << " is out of bounds for a matrix with " << numRows << " rows.", nullptr );

  return PyLong_FromLongLong( self->matrix->removeNonZeros( row,
                                                            std::get< 1 >( colsInfo ),
                                                            std::get< 2 >( colsInfo ) ) );
}

static constexpr char const * PyCRSMatrix_addToRowDocString =
"";
static PyObject * PyCRSMatrix_addToRow( PyCRSMatrix * const self, PyObject * const args )
{
  VERIFY_NON_NULL_SELF( self );
  VERIFY_INITIALIZED( self );
  VERIFY_MODIFIABLE( self );

  long row;
  PyObject * cols, * vals;
  if ( !PyArg_ParseTuple( args, "lOO", &row, &cols, &vals ) )
  { return nullptr; }

  std::tuple< PyObjectRef< PyObject >, void const *, std::ptrdiff_t > colsInfo =
    parseNumPyArray( cols, self->matrix->columnType() );

  std::tuple< PyObjectRef< PyObject >, void const *, std::ptrdiff_t > valsInfo =
    parseNumPyArray( vals, self->matrix->entryType() );

  if ( std::get< 0 >( colsInfo ) == nullptr ||  std::get< 0 >( valsInfo ) == nullptr )
  { return nullptr; }

  std::ptrdiff_t const numRows = self->matrix->numRows();
  PYTHON_ERROR_IF( row < 0 || row >= numRows, PyExc_RuntimeError,
                   "Row " << row << " is out of bounds for a matrix with " << numRows << " rows.", nullptr );

  self->matrix->addToRow( row,
                          std::get< 1 >( colsInfo ),
                          std::get< 1 >( valsInfo ),
                          std::get< 2 >( valsInfo ) );

  Py_RETURN_NONE;
}


BEGIN_ALLOW_DESIGNATED_INITIALIZERS

static PyMethodDef PyCRSMatrix_methods[] = {
  { "num_rows", (PyCFunction) PyCRSMatrix_numRows, METH_NOARGS, PyCRSMatrix_numRowsDocString },
  { "num_columns", (PyCFunction) PyCRSMatrix_numColumns, METH_NOARGS, PyCRSMatrix_numColumnsDocString },
  { "get_columns", (PyCFunction) PyCRSMatrix_getColumns, METH_VARARGS, PyCRSMatrix_getColumnsDocString },
  { "get_entries", (PyCFunction) PyCRSMatrix_getEntries, METH_VARARGS, PyCRSMatrix_getEntriesDocString },
  { "to_scipy", (PyCFunction) PyCRSMatrix_toSciPy, METH_NOARGS, PyCRSMatrix_toSciPyDocString },
  { "resize", (PyCFunction) PyCRSMatrix_resize, METH_VARARGS, PyCRSMatrix_resizeDocString },
  { "compress", (PyCFunction) PyCRSMatrix_compress, METH_NOARGS, PyCRSMatrix_compressDocString },
  { "insert_nonzeros", (PyCFunction) PyCRSMatrix_insertNonZeros, METH_VARARGS, PyCRSMatrix_insertNonZerosDocString },
  { "remove_nonzeros", (PyCFunction) PyCRSMatrix_removeNonZeros, METH_VARARGS, PyCRSMatrix_removeNonZerosDocString },
  { "add_to_row", (PyCFunction) PyCRSMatrix_addToRow, METH_VARARGS, PyCRSMatrix_addToRowDocString },
  { nullptr, nullptr, 0, nullptr } // Sentinel
};

static PyTypeObject PyCRSMatrixType = {
  PyVarObject_HEAD_INIT( nullptr, 0 )
  .tp_name = "LvArray.CRSMatrix",
  .tp_basicsize = sizeof( PyCRSMatrix ),
  .tp_itemsize = 0,
  .tp_dealloc = (destructor) PyCRSMatrix_dealloc,
  .tp_repr = PyCRSMatrix_repr,
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc = PyCRSMatrix::docString,
  .tp_methods = PyCRSMatrix_methods,
  .tp_new = PyType_GenericNew,
};

END_ALLOW_DESIGNATED_INITIALIZERS

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyTypeObject * getPyCRSMatrixType()
{ return &PyCRSMatrixType; }

namespace internal
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyObject * create( std::unique_ptr< PyCRSMatrixWrapperBase > && matrix )
{
  // Create a new Group and set the dataRepository::Group it points to.
  PyObject * const ret = PyObject_CallFunction( reinterpret_cast< PyObject * >( getPyCRSMatrixType() ), "" );
  PyCRSMatrix * const retMatrix = reinterpret_cast< PyCRSMatrix * >( ret );
  if ( retMatrix == nullptr )
  { return nullptr; }

  retMatrix->matrix = matrix.release();

  return ret;
}

} // namespace internal

} // namespace python
} // namespace LvArray
