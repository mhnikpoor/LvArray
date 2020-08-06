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
class PyArrayWrapperBase
{
public:

  /**
   *
   */
  PyArrayWrapperBase( bool const modifiable ):
    m_modifiable( modifiable )
  {}

  /**
   *
   */
  virtual ~PyArrayWrapperBase() = default;

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
  virtual std::type_index indexType() const = 0;

  /**
   *
   */
  virtual int ndim() const = 0;

  /**
   *
   */
  virtual int getSingleParameterResizeIndex() const = 0;

  /**
   *
   */
  virtual void setSingleParameterResizeIndex( int const dim ) const = 0;

  /**
   *
   */
  virtual void resize( std::ptrdiff_t const newSize ) = 0;

  /**
   *
   */
  virtual void resize( std::ptrdiff_t const * const newSizes ) = 0;

  /**
   *
   */
  virtual PyObject * toNumPy() = 0;

protected:
  bool const m_modifiable;
};

/**
 *
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
   *
   */
  PyArrayWrapper( Array< T, NDIM, PERM, INDEX_TYPE, BUFFER_TYPE > & array, bool const modify ):
    PyArrayWrapperBase( modify ),
    m_array( array )
  {}

  /**
   *
   */
  virtual ~PyArrayWrapper() = default;

  /**
   *
   */
  virtual std::string repr() const final override
  { return system::demangleType< Array< T, NDIM, PERM, INDEX_TYPE, BUFFER_TYPE > >(); }

  /**
   *
   */
  virtual std::type_index indexType() const final override
  { return std::type_index( typeid( INDEX_TYPE ) ); }

  /**
   *
   */
  virtual int ndim() const final override
  { return NDIM; }

  /**
   *
   */
  virtual int getSingleParameterResizeIndex() const final override
  { return m_array.getSingleParameterResizeIndex(); }

  /**
   *
   */
  virtual void setSingleParameterResizeIndex( int const dim ) const final override
  { m_array.setSingleParameterResizeIndex( dim ); }

  /**
   *
   */
  virtual void resize( std::ptrdiff_t const newSize ) final override
  { m_array.resize( integerConversion< INDEX_TYPE >( newSize ) ); }

  /**
   *
   */
  virtual void resize( std::ptrdiff_t const * const newSizes ) final override
  {
    INDEX_TYPE dims[ NDIM ];

    for ( int i = 0; i < NDIM; ++i )
    { dims[ i ] = integerConversion< INDEX_TYPE >( newSizes[ i ] ); }

    m_array.resize( NDIM, dims );
  }

  /**
   *
   */
  virtual PyObject * toNumPy() final override
  { return toNumPyImpl(); };

private:

  template< typename _T=T >
  std::enable_if_t< !std::is_same< _T, std::string >::value, PyObject * >
  toNumPyImpl()
  { return createNumPyArray( m_array.data(), m_modifiable, NDIM, m_array.dims(), m_array.strides() ); }

  template< typename _T=T >
  std::enable_if_t< std::is_same< _T, std::string >::value, PyObject * >
  toNumPyImpl()
  {
    LVARRAY_ERROR( "Not yet implemented." );
    return nullptr;
  }

  Array< T, NDIM, PERM, INDEX_TYPE, BUFFER_TYPE > & m_array;
};

/**
 *
 */
PyObject * create( std::unique_ptr< internal::PyArrayWrapperBase > && array );

} // namespace internal

/**
 *
 */
template< typename T,
          int NDIM,
          typename PERM,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
PyObject * create( Array< T, NDIM, PERM, INDEX_TYPE, BUFFER_TYPE > & array, bool const modify )
{
  array.move( MemorySpace::CPU, modify );
  return internal::create( std::make_unique< internal::PyArrayWrapper< T, NDIM, PERM, INDEX_TYPE, BUFFER_TYPE > >( array, modify ) );
}

/**
 *
 */
PyTypeObject * getPyArrayType();

} // namespace python
} // namespace LvArray
