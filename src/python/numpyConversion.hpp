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

/**
 * @file numpyConversion.hpp
 * @brief Contains methods to help with conversion to python objects.
 */

#pragma once

// Source include
#include "../IntegerConversion.hpp"

// system includes
#include <vector>
#include <typeindex>
#include <type_traits>

// Forward declaration of PyObject. Taken from
// https://mail.python.org/pipermail/python-dev/2003-August/037601.html
#ifndef PyObject_HEAD
struct _object;
typedef _object PyObject;
#endif

namespace LvArray
{
namespace python
{
namespace internal
{

PyObject * createNumpyArrayImpl( void * const data,
                                 std::type_index const type,
                                 bool const dataIsConst,
                                 int const ndim,
                                 std::ptrdiff_t const * const dims,
                                 std::ptrdiff_t const * const strides );

} // namespace internal

template< typename T, typename INDEX_TYPE >
std::enable_if_t< std::is_arithmetic< T >::value, PyObject * >
createNumPyArray( T * const data,
                  bool const modify,
                  int const ndim,
                  INDEX_TYPE const * const dimsPtr,
                  INDEX_TYPE const * const stridesPtr )
{
  std::vector< std::ptrdiff_t > dims( ndim );
  std::vector< std::ptrdiff_t > strides( ndim );

  for ( int i = 0; i < ndim; ++i )
  {
    dims[ i ] = integerConversion< std::ptrdiff_t >( dimsPtr[ i ] );
    strides[ i ] = integerConversion< std::ptrdiff_t >( stridesPtr[ i ] );
  }

  return internal::createNumpyArrayImpl( const_cast< void * >( static_cast< void const * >( data ) ),
                                         std::type_index( typeid( T ) ),
                                         std::is_const< T >::value || !modify,
                                         ndim,
                                         dims.data(),
                                         strides.data() );
}

template< typename T >
std::enable_if_t< std::is_arithmetic< T >::value, PyObject * >
create( T & value, bool const modify )
{
  std::ptrdiff_t dims = 1;
  std::ptrdiff_t strides = 1;

  return internal::createNumpyArrayImpl( const_cast< void * >( static_cast< void const * >( &value ) ),
                                         std::type_index( typeid( T ) ),
                                         std::is_const< T >::value || !modify,
                                         1,
                                         &dims,
                                         &strides );
}

} // namespace python
} // namespace LvArray
