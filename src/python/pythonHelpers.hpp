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

#pragma once

// Source includes
#include "pythonForwardDeclarations.hpp"
#include "../system.hpp"
#include "../Macros.hpp"

#if defined(PyObject_HEAD)
  #define PYTHON_ERROR_IF( CONDITION, TYPE, MSG, RET ) \
    do \
    { \
      if ( CONDITION ) \
      { \
        std::ostringstream __oss; \
        __oss << "***** ERROR\n"; \
        __oss << "***** LOCATION: " LOCATION "\n"; \
        __oss << "***** Controlling expression (should be false): " STRINGIZE( EXP ) "\n"; \
        __oss << MSG << "\n"; \
        __oss << LvArray::system::stackTrace(); \
        PyErr_SetString( TYPE, __oss.str().c_str() ); \
        return RET; \
      } \
    } while( false )
#else
  #define PYTHON_ERROR_IF( CONDITION, TYPE, MSG, RET ) \
    static_assert( false, "You are attempting to use PYTHON_ERROR_IF but haven't yet included Python.hpp" )
#endif


namespace LvArray
{
namespace python
{
namespace internal
{

/**
 *
 */
void xincref( void * obj );

/**
 *
 */
void xdecref( void * obj );

/**
 *
 */
bool canConvert( PyObject * const obj, PyTypeObject * type );

} // namespace internal

/**
 *
 */
template< typename T = PyObject >
class PyObjectRef
{
public:

  /**
   * @brief Create an uninitialized (nullptr) reference.
   */
  PyObjectRef() = default;

  /**
   * @brief Take ownership of a reference to @p src.
   * @p src The object to be referenced.
   */
  PyObjectRef( T * const src ):
    m_object( src )
  {}

  /**
   *
   */
  PyObjectRef( PyObjectRef const & src )
  { *this = src; }
  
  /**
   *
   */
  PyObjectRef( PyObjectRef && src )
  { *this = std::move( src ); }

  /**
   *
   */
  ~PyObjectRef()
  { internal::xdecref( m_object ); }

  /**
   *
   */
  PyObjectRef & operator=( PyObjectRef const & src )
  {
    m_object = src.m_object;
    internal::xincref( src.m_object );
    return *this;
  }

  /**
   *
   */
  PyObjectRef & operator=( PyObjectRef && src )
  {
    m_object = src.m_object;
    src.m_object = nullptr;
    return *this;
  }

  /**
   * @brief Decrease the reference count to the current object and take ownership
   *   of a new reference.
   * @p src The new object to be referenced.
   * @return *this.
   */
  PyObjectRef & operator=( PyObject * src )
  {
    internal::xdecref( m_object );

    m_object = src;
    return *this;
  }

  /**
   *
   */
  operator T*()
  { return m_object; }

  /**
   *
   */
  T * get() const
  { return m_object; }

  /**
   *
   */
  T ** getAddress()
  { return &m_object; }

  /**
   *
   */
  T * release()
  { 
    T * const ret = m_object;
    m_object = nullptr;
    return ret;
  }

private:
  T * m_object = nullptr;
};

/**
 *
 */
template< typename T >
T * convert( PyObject * const obj, PyTypeObject * type )
{
  if( internal::canConvert( obj, type ) )
  { return reinterpret_cast< T * >( obj ); }

  return nullptr;
}

} // namespace python
} // namespace LvArray
