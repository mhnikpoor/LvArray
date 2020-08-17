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
 * @file
 */

#pragma once

// Source includes
#include "pythonForwardDeclarations.hpp"
#include "numpyHelpers.hpp"
#include "../ArrayOfArrays.hpp"
#include "../Macros.hpp"
#include "../limits.hpp"
#include "../output.hpp"

// System includes
#include <string>
#include <memory>
#include <typeindex>

namespace LvArray
{
namespace python
{
namespace internal
{

/**
 *
 */
class PyArrayOfArraysWrapperBase
{
public:

  /**
   *
   */
  PyArrayOfArraysWrapperBase( bool const modifiable ):
    m_modifiable( modifiable )
  {}

  /**
   *
   */
  virtual ~PyArrayOfArraysWrapperBase() = default;

  /**
   *
   */
  bool modifiable() const
  { return m_modifiable; }

  /**
   *
   */
  virtual std::string repr() const = 0;

  /**
   *
   */
  virtual long long size() const = 0;

  /**
   *
   */
  virtual PyObject * operator[]( long long index ) = 0;


protected:
  bool const m_modifiable;
};

/**
 *
 */
template< typename T,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class PyArrayOfArraysWrapper : public PyArrayOfArraysWrapperBase
{
public:

  /**
   *
   */
  PyArrayOfArraysWrapper( ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE > & arrayOfArrays, bool const modify ):
    PyArrayOfArraysWrapperBase( modify ),
    m_arrayOfArrays( arrayOfArrays )
  {}

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual ~PyArrayOfArraysWrapper() = default;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::string repr() const final override
  { return system::demangleType< ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE > >(); }

  /**
   *
   */
  virtual long long size() const final override
  {
    INDEX_TYPE numArrays = m_arrayOfArrays.size();

    long long foobar = integerConversion< long long >( numArrays );
    printf("%ld %lld", numArrays, foobar);
    return integerConversion< long long >( numArrays );
  }

  /**
   *
   */
  virtual PyObject * operator[]( long long index ) override{
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( index );
    ArraySlice< T, 1, 0, INDEX_TYPE > slice = m_arrayOfArrays[ convertedIndex ];
    T* data = slice;
    constexpr INDEX_TYPE strides = 1;
    INDEX_TYPE size = slice.size();
    return createNumPyArray( data, m_modifiable, 1, &size, &strides );
  }

private:
  ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE > & m_arrayOfArrays;
};

/**
 *
 */
PyObject * create( std::unique_ptr< internal::PyArrayOfArraysWrapperBase > && array );

} // namespace internal

/**
 *
 */
template< typename T, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
PyObject * create( ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE > & array, bool const modify )
{
  array.move( MemorySpace::CPU, modify );
  return internal::create( std::make_unique< internal::PyArrayOfArraysWrapper< T, INDEX_TYPE, BUFFER_TYPE > >( array, modify ) );
}


/**
 *
 */
PyTypeObject * getPyArrayOfArraysType();

} // namespace python
} // namespace LvArray
