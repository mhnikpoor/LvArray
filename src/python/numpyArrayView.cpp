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
 * @file numpyArrayView.hpp
 * @brief Contains functions to create a NumPy ndarray from an ArrayView.
 */
#define PY_SSIZE_T_CLEAN

#include <Python.h>

// source includes
#include "numpyArrayView.hpp"

namespace LvArray
{
namespace python
{
namespace internal
{

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
PyObject * createPyListOfStrings( std::string const * const strptr, std::ptrdiff_t const size )
{
  PyObject * pylist = PyList_New( size );
  for ( std::ptrdiff_t i = 0; i < size; ++i )
  {
    PyObject * pystr = PyUnicode_FromString( strptr[ i ].c_str() );
    PyList_SET_ITEM( pylist, integerConversion< Py_ssize_t >( i ), pystr );
  }
  return pylist;
}

} // namespace internal
} // namespace python
} // namespace LvArray
