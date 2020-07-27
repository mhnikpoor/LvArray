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

/**
 * @brief create and return a Python list of strings from a std::vector of std::strings.
 *        the Python strings will be copies.
 * @param vec the vector to convert.
 */
PyObject * create( std::vector< std::string > vec, bool const modify )
{
    LVARRAY_UNUSED_VARIABLE( modify );
    return internal::createPyListOfStrings( vec.data(), integerConversion< std::ptrdiff_t >( vec.size() ) );
}

namespace internal
{

/**
 * @brief create and return a Python list of strings from an array of std::strings.
 *        the Python strings will be copies.
 * @param strptr a pointer to the strings to convert
 * @param size the number of strings in the array
 */
PyObject * createPyListOfStrings( std::string * strptr, std::ptrdiff_t size ){
    PyObject * pystr;
    PyObject * pylist = PyList_New( size );
    for ( std::ptrdiff_t i = 0; i < size; ++i )
    {
         pystr = PyUnicode_FromString( strptr[ i ].c_str() );
         PyList_SET_ITEM( pylist, integerConversion< Py_ssize_t >( i ), pystr );
    }
    return pylist;
}

} // namespace internal
} // namespace python
} // namespace LvArray
