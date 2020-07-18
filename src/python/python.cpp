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
#include "python.hpp"
#include "../limits.hpp"
#include "pythonHelpers.hpp"

namespace LvArray
{
namespace python
{

/**
 *
 */
static PyMethodDef LvArrayFuncs[] = {
  { nullptr, nullptr, 0, nullptr }        /* Sentinel */
};

static constexpr char const * LvArrayDocString =
  "Manipulate LvArray objects from Python.\n\n"
  "This module provides classes for manipulating LvArray C++ classes from Python.\n"
  "Instances of the Python classes may not be constructed in Python;\n"
  "an existing LvArray object must be\n"
  "created in C++ and then converted to one of these Python classes."
;

/**
 * Initialize the module object for Python with the exported functions
 */
static struct PyModuleDef LvArrayModuleFunctions = {
  PyModuleDef_HEAD_INIT,
  .m_name = "LvArray",
  .m_doc = LvArrayDocString,
  .m_size = -1,
  .m_methods = LvArrayFuncs
};

static bool addConstants( PyObject * module )
{
  std::array< std::pair< long, char const * >, 3 > const constants = { {
    { static_cast< long >( LvArray::python::PyModify::READ_ONLY ), "READ_ONLY" },
    { static_cast< long >( LvArray::python::PyModify::MODIFIABLE ), "MODIFIABLE" },
    { static_cast< long >( LvArray::python::PyModify::RESIZEABLE ), "RESIZEABLE" }
  } };

  for( std::pair< long, char const * > const & pair : constants )
  {
    PYTHON_ERROR_IF( PyModule_AddIntConstant( module, pair.second, pair.first ), PyExc_RuntimeError,
                     "couldn't add constant", false );
  }

  PYTHON_ERROR_IF( PyModule_AddIntConstant( module, "CPU", static_cast< long >( LvArray::MemorySpace::CPU ) ), PyExc_RuntimeError,
                   "couldn't add constant", false );

  #if defined(USE_CUDA)
  PYTHON_ERROR_IF( PyModule_AddIntConstant( module, "GPU", static_cast< long >( LvArray::MemorySpace::GPU ) ), PyExc_RuntimeError,
                   "couldn't add constant", false );
  #endif

  return true;
}

PyObjectRef<> getModule()
{
  // import_array();

  LvArray::python::PyObjectRef<> module{ PyModule_Create( &LvArrayModuleFunctions ) };
  if( module == nullptr )
  {
    return nullptr;
  }

  if( !LvArray::python::addTypeToModule( module, LvArray::python::getPyArrayType(), "Array" ) )
  {
    return nullptr;
  }

  if( !LvArray::python::addTypeToModule( module, LvArray::python::getPySortedArrayType(), "SortedArray" ) )
  {
    return nullptr;
  }

  if( !LvArray::python::addTypeToModule( module, LvArray::python::getPyArrayOfArraysType(), "ArrayOfArrays" ) )
  {
    return nullptr;
  }

  if( !LvArray::python::addTypeToModule( module, LvArray::python::getPyArrayOfSetsType(), "ArrayOfSets" ) )
  {
    return nullptr;
  }

  if( !LvArray::python::addTypeToModule( module, LvArray::python::getPyCRSMatrixType(), "CRSMatrix" ) )
  {
    return nullptr;
  }

  if( !addConstants( module ) )
  {
    return nullptr;
  }

  // Since we return module we don't want to decrease the reference count.
  return module.release();
}

bool addPyLvArrayModule( PyObject * module )
{
  LvArray::python::PyObjectRef<> pylvarrayModule { PyImport_ImportModule( "pylvarray" ) };
  if( PyModule_AddObject( module, "pylvarray", pylvarrayModule ) < 0 )
  {
    return false;
  }
  return true;
}

} // namespace python
} // namespace LvArray

PyMODINIT_FUNC
PyInit_pylvarray( void )
{
  return LvArray::python::getModule().release();
}
