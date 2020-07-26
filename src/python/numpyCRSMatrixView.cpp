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
#include "numpyCRSMatrixView.hpp"

// System includes
#define NPY_NO_DEPRECATED_API NPY_1_15_API_VERSION
#include <numpy/arrayobject.h>

namespace LvArray
{
namespace python
{
namespace internal
{

PyObject * createCRSMatrix( PyObject * entries, PyObject * columns, PyObject * offsets ){
    if ( entries == NULL || columns == NULL || offsets == NULL ){
        return NULL;
    }
    return Py_BuildValue( "(NNN)", entries, columns, offsets );
}

} // namespace internal
} // namespace python
} // namespace LvArray
