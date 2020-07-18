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
#include "../SortedArray.hpp"
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
class PySortedArrayWrapperBase
{
public:

  /**
   *
   */
  PySortedArrayWrapperBase( ):
    m_accessLevel( static_cast< int >( LvArray::python::PyModify::READ_ONLY ) )
  {}

  /**
   *
   */
  virtual ~PySortedArrayWrapperBase() = default;

  /**
   * @brief Return the access level for the array.
   * @return the access level for the array.
   */
  virtual int getAccessLevel() const
  { return m_accessLevel; }

  /**
   * @brief Set the access level for the array.
   */
  virtual void setAccessLevel( int accessLevel, int memorySpace ) = 0;

  /**
   *
   */
  virtual std::string repr() const = 0;

  /**
   *
   */
  virtual std::type_index valueType() const = 0;

  /**
   *
   */
  virtual std::ptrdiff_t insert( void const * const values,
                                 std::ptrdiff_t const nVals ) = 0;

  /**
   *
   */
  virtual std::ptrdiff_t remove( void const * const values,
                                 std::ptrdiff_t const nVals ) = 0;

  /**
   *
   */
  virtual PyObject * toNumPy() = 0;

  virtual std::type_index dataType() const = 0;

protected:
  int m_accessLevel;
};

/**
 *
 */
template< typename T,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class PySortedArrayWrapper : public PySortedArrayWrapperBase
{
public:

  /**
   *
   */
  PySortedArrayWrapper( SortedArray< T, INDEX_TYPE, BUFFER_TYPE > & sortedArray ):
    PySortedArrayWrapperBase( ),
    m_sortedArray( sortedArray )
  {}

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual ~PySortedArrayWrapper() = default;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::string repr() const final override
  { return system::demangleType< SortedArray< T, INDEX_TYPE, BUFFER_TYPE > >(); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::type_index valueType() const override
  { return std::type_index( typeid( T ) ); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::ptrdiff_t insert( void const * const values,
                                 std::ptrdiff_t const nVals ) final override
  {
    T const * const castedValues = reinterpret_cast< T const * >( values );
    return integerConversion< std::ptrdiff_t >( m_sortedArray.insert( castedValues, castedValues + nVals ) );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::ptrdiff_t remove( void const * const values,
                                 std::ptrdiff_t const nVals ) final override
  {
    T const * const castedValues = reinterpret_cast< T const * >( values );
    return integerConversion< std::ptrdiff_t >( m_sortedArray.remove( castedValues, castedValues + nVals ) );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual PyObject * toNumPy() final override
  {
    INDEX_TYPE const dims = m_sortedArray.size();
    INDEX_TYPE const strides = 1;
    return createNumPyArray( m_sortedArray.data(), false, 1, &dims, &strides );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void setAccessLevel( int accessLevel, int memorySpace ) final override
  {
    LVARRAY_UNUSED_VARIABLE( memorySpace );
    if( accessLevel >= static_cast< int >( LvArray::python::PyModify::RESIZEABLE ) )
    {
      // touch
    }
    m_accessLevel = accessLevel;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::type_index dataType() const final override
  { return std::type_index( typeid( T ) ); }

private:
  SortedArray< T, INDEX_TYPE, BUFFER_TYPE > & m_sortedArray;
};

/**
 *
 */
PyObject * create( std::unique_ptr< internal::PySortedArrayWrapperBase > && array );

} // namespace internal

/**
 *
 */
template< typename T, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
std::enable_if_t< internal::canExportToNumpy< T >, PyObject * >
create( SortedArray< T, INDEX_TYPE, BUFFER_TYPE > & array )
{
  return internal::create( std::make_unique< internal::PySortedArrayWrapper< T, INDEX_TYPE, BUFFER_TYPE > >( array ) );
}

/**
 *
 */
PyTypeObject * getPySortedArrayType();

} // namespace python
} // namespace LvArray
