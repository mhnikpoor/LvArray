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
 * @file python.hpp
 * @brief Includes all the other python related headers.
 */

#pragma once

// source includes
#include "PyArray.hpp"
#include "PySortedArray.hpp"
#include "PyArrayOfArrays.hpp"
#include "PyArrayOfSets.hpp"
#include "PyCRSMatrix.hpp"
#include "PyFunc.hpp"
#include "../typeManipulation.hpp"

namespace LvArray
{

/**
 * @brief Contains all the Python code.
 */
namespace python
{

/**
 *
 */
bool addPyLvArrayModule( PyObject * module );

/**
 * @brief Expands to a static constexpr template bool @code CanCreate< T > @endcode which is true iff
 *   @c T is a type which LvArray can export to Python.
 */
IS_VALID_EXPRESSION( CanCreate, T, LvArray::python::create( std::declval< T & >() ) );

/**
 *
 */
PyObjectRef<> getModule();

} // namespace python
} // namespace LvArray
