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
#include "../Array.hpp"
#include "../Macros.hpp"
#include "../limits.hpp"
#include "../output.hpp"
#include "pythonHelpers.hpp"

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
 * @class PyArrayWrapperBase
 * @brief A virtual interface to an Array geared towards python usage.
 */
class PyArrayWrapperBase
{
public:

  /**
   * @brief Constructor.
   * Array begins as read-only
   */
  PyArrayWrapperBase():
    m_accessLevel( static_cast< int >( LvArray::python::PyModify::READ_ONLY ) )
  {}

  /**
   * @brief Virtual destructor.
   */
  virtual ~PyArrayWrapperBase() = default;

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
   * @brief Return a string representing the underlying Array type.
   * @return A string representing the underlying Array type.
   */
  virtual std::string repr() const = 0;

  /**
   * @brief Return the number of dimensions in the array.
   * @return The number of dimensions in the array.
   * @note This wraps Array::ndim.
   */
  virtual int ndim() const = 0;

  /**
   * @brief Return the single parameter resize index of the array.
   * @return the single parameter resize index of the array.
   * @note This wraps Array::getSingleParameterResizeIndex().
   */
  virtual int getSingleParameterResizeIndex() const = 0;

  /**
   * @brief Set the single parameter resize index of the array.
   * @param dim The dimension to make the single parameter resize index.
   * @note This wraps Array::setSingleParameterResizeIndex( int ).
   */
  virtual void setSingleParameterResizeIndex( int const dim ) const = 0;

  /**
   * @brief Resize the default dimension of the array.
   * @param newSize The new size of the default dimension.
   * @note This wraps Array::resize( INDEX_TYPE ).
   */
  virtual void resize( std::ptrdiff_t const newSize ) = 0;

  /**
   * @brief Resize all the dimensions of the array.
   * @param newSizes The new sizes of each dimension, must be of length @c ndim().
   * @note Ths wraps Array::resize( int, INDEX_TYPE const * )
   */
  virtual void resize( std::ptrdiff_t const * const newSizes ) = 0;

  /**
   * @brief Return a NumPy ndarray wrapping the array.
   * @details This is a shallow copy and is modifiable if @c modifiable() is @c true.
   */
  virtual PyObject * toNumPy() const = 0;

protected:

  /// If the array can be modified.
  int m_accessLevel;
};

/**
 * @class PyArrayWrapper
 * @brief An implements the PyArrayWrapperBase interface for a particular Array type.
 * @tparam T The type of values in the array.
 * @tparam NDIM The dimensionality of the array.
 * @tparam PERM The permutation of the array.
 * @tparam INDEX_TYPE The index type of the array.
 * @tparam BUFFER_TYPE The buffer type of the array.
 * @note This holds a reference to the wrapped Array, you must ensure that the reference remains valid
 *   for the lifetime of this object.
 */
template< typename T,
          int NDIM,
          typename PERM,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class PyArrayWrapper : public PyArrayWrapperBase
{
public:

  /**
   * @brief Constructor.
   * @param array The array to wrap.
   * @param modify If the array is modifiable.
   */
  PyArrayWrapper( Array< T, NDIM, PERM, INDEX_TYPE, BUFFER_TYPE > & array ):
    PyArrayWrapperBase( ),
    m_array( array )
  {}

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual ~PyArrayWrapper() = default;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::string repr() const final override
  { return system::demangleType< Array< T, NDIM, PERM, INDEX_TYPE, BUFFER_TYPE > >(); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual int ndim() const final override
  { return NDIM; }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual int getSingleParameterResizeIndex() const final override
  { return m_array.getSingleParameterResizeIndex(); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void setSingleParameterResizeIndex( int const dim ) const final override
  { m_array.setSingleParameterResizeIndex( dim ); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void resize( std::ptrdiff_t const newSize ) final override
  { m_array.resize( integerConversion< INDEX_TYPE >( newSize ) ); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void resize( std::ptrdiff_t const * const newSizes ) final override
  {
    INDEX_TYPE dims[ NDIM ];

    for ( int i = 0; i < NDIM; ++i )
    { dims[ i ] = integerConversion< INDEX_TYPE >( newSizes[ i ] ); }

    m_array.resize( NDIM, dims );
  }

  virtual void setAccessLevel( int accessLevel ) final override
  {
    if ( accessLevel >= static_cast< int >( LvArray::python::PyModify::RESIZEABLE ) ){
      // touch
    }
    m_accessLevel = accessLevel;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual PyObject * toNumPy() const final override
  { return toNumPyImpl(); };

private:

  /**
   * @brief Create a NumPy ndarray for an Array that doesn't contain std::strings.
   */
  template< typename _T=T >
  std::enable_if_t< !std::is_same< _T, std::string >::value, PyObject * >
  toNumPyImpl() const
  { return createNumPyArray( m_array.data(), m_accessLevel >= static_cast< int >( LvArray::python::PyModify::MODIFIABLE ), NDIM, m_array.dims(), m_array.strides() ); }

  /**
   * @brief Create a NumPy ndarray for an Array that contains std::strings.
   */
  template< typename _T=T >
  std::enable_if_t< std::is_same< _T, std::string >::value, PyObject * >
  toNumPyImpl() const
  { return createPyListOfStrings( m_array.data(), integerConversion< std::ptrdiff_t >( m_array.size() ) ); }

  Array< T, NDIM, PERM, INDEX_TYPE, BUFFER_TYPE > & m_array;
};

/**
 * @brief Create a Python object from a PyArrayWrapperBase.
 * @param array The Array to export to Python.
 */
PyObject * create( std::unique_ptr< internal::PyArrayWrapperBase > && array );

} // namespace internal

/**
 * @brief Create a Python object from an Array.
 * @tparam T The type of values in the array.
 * @tparam NDIM The dimensionality of the array.
 * @tparam PERM The permutation of the array.
 * @tparam INDEX_TYPE The index type of the array.
 * @tparam BUFFER_TYPE The buffer type of the array.
 * @param array The Array to export to Python.
 * @param modify If the array is modifiable.
 * @note @p array is moved to the CPU and touched if @p modify is @c true.
 */
template< typename T,
          int NDIM,
          typename PERM,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
PyObject * create( Array< T, NDIM, PERM, INDEX_TYPE, BUFFER_TYPE > & array )
{
  using WrapperType = internal::PyArrayWrapper< T, NDIM, PERM, INDEX_TYPE, BUFFER_TYPE >;
  return internal::create( std::make_unique< WrapperType >( array ) );
}

/**
 * @brief Return the Python type for the Array.
 */
PyTypeObject * getPyArrayType();

} // namespace python
} // namespace LvArray
