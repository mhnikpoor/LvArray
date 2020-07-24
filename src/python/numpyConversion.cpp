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
#include "numpyConversion.hpp"
#include "../StringUtilities.hpp"

// System includes
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#define NPY_NO_DEPRECATED_API NPY_1_15_API_VERSION
#include <numpy/arrayobject.h>
#pragma GCC diagnostic pop

namespace LvArray
{
namespace python
{
namespace internal
{

std::pair< int, std::size_t > getNumPyType( std::type_index const typeIndex )
{
  if( typeIndex == std::type_index( typeid( char ) ) )
  { return { std::is_signed< char >::value ? NPY_BYTE : NPY_UBYTE, sizeof( char ) }; }
  if( typeIndex == std::type_index( typeid( signed char ) ) )
  { return { NPY_BYTE, sizeof( signed char ) }; }
  if( typeIndex == std::type_index( typeid( unsigned char ) ) )
  { return { NPY_UBYTE, sizeof( unsigned char ) }; }
  if( typeIndex == std::type_index( typeid( short ) ) )
  { return { NPY_SHORT, sizeof( short ) }; }
  if( typeIndex == std::type_index( typeid( unsigned short ) ) )
  { return { NPY_USHORT, sizeof( unsigned short ) }; }
  if( typeIndex == std::type_index( typeid( int ) ) )
  { return { NPY_INT, sizeof( int ) }; }
  if( typeIndex == std::type_index( typeid( unsigned int ) ) )
  { return { NPY_UINT, sizeof( unsigned int ) }; }
  if( typeIndex == std::type_index( typeid( long ) ) )
  { return { NPY_LONG, sizeof( long ) }; }
  if( typeIndex == std::type_index( typeid( unsigned long ) ) )
  { return { NPY_ULONG, sizeof( unsigned long ) }; }
  if( typeIndex == std::type_index( typeid( long long ) ) )
  { return { NPY_LONGLONG, sizeof( long long ) }; }
  if( typeIndex == std::type_index( typeid( unsigned long long ) ) )
  { return { NPY_ULONGLONG, sizeof( unsigned long long ) }; }
  if( typeIndex == std::type_index( typeid( float ) ) )
  { return { NPY_FLOAT, sizeof( float ) }; }
  if( typeIndex == std::type_index( typeid( double ) ) )
  { return { NPY_DOUBLE, sizeof( double ) }; }
  if( typeIndex == std::type_index( typeid( long double ) ) )
  { return { NPY_LONGDOUBLE, sizeof( long double ) }; }

  LVARRAY_ERROR( "No NumpyType for " << demangle( typeIndex.name() ) );
  return { NPY_NOTYPE, std::numeric_limits< std::size_t >::max() };
}

PyObject * createNumpyArrayImpl( void * const data,
                                 std::type_index const type,
                                 bool const dataIsConst,
                                 int const ndim,
                                 std::ptrdiff_t const * const dims,
                                 std::ptrdiff_t const * const strides )
{
  if( PyArray_API == nullptr )
  { import_array(); }

  std::pair< int, std::size_t > const typeInfo = getNumPyType( type );

  std::vector< npy_intp > byteStrides( ndim );
  std::vector< npy_intp > npyDims( ndim );

  for ( int i = 0; i < ndim; ++i )
  {
    byteStrides[ i ] = integerConversion< npy_intp >( strides[ i ] * typeInfo.second );
    npyDims[ i ] = integerConversion< npy_intp >( dims[ i ] );
  }

  int const flags = dataIsConst ? 0 : NPY_ARRAY_WRITEABLE;
  return PyArray_NewFromDescr( &PyArray_Type,
                               PyArray_DescrFromType( typeInfo.first ),
                               ndim,
                               npyDims.data(),
                               byteStrides.data(),
                               data,
                               flags,
                               NULL );
}

} // namespace internal
} // namespace python
} // namespace LvArray
