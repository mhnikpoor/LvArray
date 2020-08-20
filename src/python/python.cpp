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
"";

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


PyObjectRef<> getModule()
{
  // import_array();

  LvArray::python::PyObjectRef<> module{ PyModule_Create( &LvArrayModuleFunctions ) };
  if ( module == nullptr )
  { return nullptr; }

  if ( !LvArray::python::addTypeToModule( module, LvArray::python::getPyArrayType(), "Array" ) )
  { return nullptr; }

  if ( !LvArray::python::addTypeToModule( module, LvArray::python::getPySortedArrayType(), "SortedArray" ) )
  { return nullptr; }

  if ( !LvArray::python::addTypeToModule( module, LvArray::python::getPyArrayOfArraysType(), "ArrayOfArrays" ) )
  { return nullptr; }

  if ( !LvArray::python::addTypeToModule( module, LvArray::python::getPyCRSMatrixType(), "CRSMatrix" ) )
  { return nullptr; }

  // Since we return module we don't want to decrease the reference count.
  return module;
}

} // namespace python
} // namespace LvArray
