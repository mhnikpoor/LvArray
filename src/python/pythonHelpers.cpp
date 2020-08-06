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
#include "pythonHelpers.hpp"

namespace LvArray
{
namespace python
{
namespace internal
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void xincref( void * obj )
{ Py_XINCREF( obj ); }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void xdecref( void * obj )
{ Py_XDECREF( obj ); }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool canConvert( PyObject * const obj, PyTypeObject * type )
{
  PYTHON_ERROR_IF( obj == nullptr, PyExc_RuntimeError, "Passed a nullptr as an argument.", false );

  int isInstanceOfType = PyObject_IsInstance( obj, reinterpret_cast< PyObject * >( type ) );
  if ( isInstanceOfType < 0 )
  { return false; }

  PYTHON_ERROR_IF( !isInstanceOfType, PyExc_RuntimeError, "Expect an argument of type " << type->tp_name, false );

  return true;
}

} // namespace internal
} // namespace python
} // namespace LvArray
