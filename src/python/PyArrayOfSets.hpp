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
#include "../ArrayOfSets.hpp"
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
class PyArrayOfSetsWrapperBase
{
public:

  /**
   *
   */
  PyArrayOfSetsWrapperBase( ):
    m_accessLevel( static_cast< int >( LvArray::python::PyModify::READ_ONLY ) )
  {}

  /**
   *
   */
  virtual ~PyArrayOfSetsWrapperBase() = default;

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
  virtual PyObject * operator[]( long long index ) = 0;

  /**
   *
   */
  virtual std::type_index valueType() const = 0;

  /**
   *
   */
  virtual void eraseSet( long long index ) = 0;

  /**
   *
   */
  virtual void removeFromSet( long long index, void const * data, long long numvals ) = 0;

  /**
   *
   */
  virtual void insertSet( long long index, long long capacity ) = 0;

  /**
   *
   */
  virtual void insertIntoSet( long long setIndex, void const * data, std::ptrdiff_t numvals ) = 0;

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
class PyArrayOfSetsWrapper : public PyArrayOfSetsWrapperBase
{
public:

  /**
   *
   */
  PyArrayOfSetsWrapper( ArrayOfSets< T, INDEX_TYPE, BUFFER_TYPE > & arrayOfSets ):
    PyArrayOfSetsWrapperBase( ),
    m_arrayOfSets( arrayOfSets )
  {}

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual ~PyArrayOfSetsWrapper() = default;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::string repr() const final override
  { return system::demangleType< ArrayOfSets< T, INDEX_TYPE, BUFFER_TYPE > >(); }

  /**
   *
   */
  virtual long long size() const final override
  {
    return integerConversion< long long >( m_arrayOfSets.size() );
  }

  /**
   *
   */
  virtual PyObject * operator[]( long long index ) override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( index );
    ArraySlice< T const, 1, 0, INDEX_TYPE > slice = m_arrayOfSets[ convertedIndex ];
    T const * data = slice;
    constexpr INDEX_TYPE strides = 1;
    INDEX_TYPE size = slice.size();
    return createNumPyArray( data, false, 1, &size, &strides );
  }

  /**
   *
   */
  virtual void eraseSet( long long index ) override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( index );
    m_arrayOfSets.eraseSet( convertedIndex );
  }

  /**
   *
   */
  virtual void removeFromSet( long long index, void const * data, long long numvals ) override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( index );
    T const * begin = static_cast< T const * >( data );
    T const * end = begin + numvals;
    m_arrayOfSets.removeFromSet( convertedIndex, begin, end );
  }

  /**
   *
   */
  virtual void insertSet( long long index, long long capacity ) override
  {
    INDEX_TYPE convertedIndex = integerConversion< INDEX_TYPE >( index );
    INDEX_TYPE convertedCapacity = integerConversion< INDEX_TYPE >( capacity );
    m_arrayOfSets.insertSet( convertedIndex, convertedCapacity );
  }

  /**
   *
   */
  virtual void insertIntoSet( long long setIndex, void const * data, std::ptrdiff_t numvals ) override
  {
    INDEX_TYPE convertedSetIndex = integerConversion< INDEX_TYPE >( setIndex );
    T const * begin = static_cast< T const * >( data );
    T const * end = begin + numvals;
    m_arrayOfSets.insertIntoSet( convertedSetIndex, begin, end );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::type_index valueType() const override
  { return std::type_index( typeid( T ) ); }

  virtual void setAccessLevel( int accessLevel ) final override
  {
    if ( accessLevel >= static_cast< int >( LvArray::python::PyModify::RESIZEABLE ) ){
      // touch
    }
    m_accessLevel = accessLevel;
  }

private:
  ArrayOfSets< T, INDEX_TYPE, BUFFER_TYPE > & m_arrayOfSets;
};

/**
 *
 */
PyObject * create( std::unique_ptr< internal::PyArrayOfSetsWrapperBase > && array );

} // namespace internal

/**
 *
 */
template< typename T, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
PyObject * create( ArrayOfSets< T, INDEX_TYPE, BUFFER_TYPE > & array )
{
  return internal::create( std::make_unique< internal::PyArrayOfSetsWrapper< T, INDEX_TYPE, BUFFER_TYPE > >( array ) );
}


/**
 *
 */
PyTypeObject * getPyArrayOfSetsType();

} // namespace python
} // namespace LvArray
