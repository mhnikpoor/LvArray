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
  PyArrayOfArraysWrapperBase( ):
    m_accessLevel( static_cast< int >( LvArray::python::PyModify::READ_ONLY ) )
  {}

  /**
   *
   */
  virtual ~PyArrayOfArraysWrapperBase() = default;

  /**
   * @brief Return the access level for the array.
   * @return the access level for the array.
   */
  virtual int getAccessLevel() const
  { return m_accessLevel; }

  /**
   * @brief Set the access level for the array.
   */
  virtual void setAccessLevel( int accessLevel ) = 0;

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
  virtual long long sizeOfArray( long long index ) const = 0;

  /**
   *
   */
  virtual PyObject * operator[]( long long index ) = 0;

  /**
   *
   */
  virtual std::type_index valueType() const = 0;

  /**
   *
   */
  virtual void eraseArray( long long index ) = 0;

  /**
   *
   */
  virtual void eraseFromArray( long long index, long long begin ) = 0;

  /**
   *
   */
  virtual void insertArray( long long index, void const * data, std::ptrdiff_t numvals ) = 0;

  /**
   *
   */
  virtual void insertIntoArray( long long array, long long index, void const * data, std::ptrdiff_t numvals ) = 0;

protected:
  /// access level for the array
  int m_accessLevel;
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
  PyArrayOfArraysWrapper( ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE > & arrayOfArrays ):
    PyArrayOfArraysWrapperBase( ),
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
    return integerConversion< long long >( m_arrayOfArrays.size() );
  }

  /**
   *
   */
  virtual long long sizeOfArray( long long index ) const final override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( index );
    return integerConversion< long long >( m_arrayOfArrays.sizeOfArray( convertedIndex ) );
  }

  /**
   *
   */
  virtual PyObject * operator[]( long long index ) override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( index );
    ArraySlice< T, 1, 0, INDEX_TYPE > slice = m_arrayOfArrays[ convertedIndex ];
    T * data = slice;
    constexpr INDEX_TYPE strides = 1;
    INDEX_TYPE size = slice.size();
    return createNumPyArray( data, getAccessLevel() >= static_cast< int >( LvArray::python::PyModify::MODIFIABLE ), 1, &size, &strides );
  }

  /**
   *
   */
  virtual void eraseArray( long long index ) override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( index );
    m_arrayOfArrays.eraseArray( convertedIndex );
  }

  /**
   *
   */
  virtual void eraseFromArray( long long index, long long begin ) override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( index );
    INDEX_TYPE convertedBegin = integerConversion< INDEX_TYPE >( begin );
    m_arrayOfArrays.eraseFromArray( convertedIndex, convertedBegin );
  }

  /**
   *
   */
  virtual void insertArray( long long index, void const * data, std::ptrdiff_t numvals ) override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( index );
    T const * begin = static_cast< T const * >( data );
    T const * end = begin + numvals;
    m_arrayOfArrays.insertArray( convertedIndex, begin, end );
  }

  /**
   *
   */
  virtual void insertIntoArray( long long array, long long index, void const * data, std::ptrdiff_t numvals ) override
  {
    INDEX_TYPE convertedArray = integerConversion< INDEX_TYPE >( array );
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( index );
    T const * begin = static_cast< T const * >( data );
    T const * end = begin + numvals;
    m_arrayOfArrays.insertIntoArray( convertedArray, convertedIndex, begin, end );
  }

  virtual void setAccessLevel( int accessLevel ) final override
  {
    if( accessLevel >= static_cast< int >( LvArray::python::PyModify::RESIZEABLE ) )
    {
      // touch
    }
    m_accessLevel = accessLevel;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::type_index valueType() const override
  { return std::type_index( typeid( T ) ); }

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
std::enable_if_t< internal::canExportToNumpy< T >, PyObject * >
create( ArrayOfArrays< T, INDEX_TYPE, BUFFER_TYPE > & array )
{
  return internal::create( std::make_unique< internal::PyArrayOfArraysWrapper< T, INDEX_TYPE, BUFFER_TYPE > >( array ) );
}


/**
 *
 */
PyTypeObject * getPyArrayOfArraysType();

} // namespace python
} // namespace LvArray
