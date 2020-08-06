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

#pragma once

// Source includes
#include "numpyHelpers.hpp"
#include "../ArrayView.hpp"
#include "../limits.hpp"

// System includes
#include <vector>

namespace LvArray
{

namespace python
{

namespace internal
{

/**
 * @brief create and return a Python list of strings from an array of std::strings.
 *   the Python strings will be copies.
 * @param strptr a pointer to the strings to convert
 * @param size the number of strings in the array
 */
PyObject * createPyListOfStrings( std::string const * const strptr, std::ptrdiff_t const size );

} // namespace internal

/**
 * @brief create and return a Python list of strings from a std::vector of std::strings.
 *   the Python strings will be copies.
 * @param vec the vector to convert.
 * @param modify has no effect
 */
inline PyObject * create( std::vector< std::string > const & vec, bool const modify )
{
  LVARRAY_UNUSED_VARIABLE( modify );
  return internal::createPyListOfStrings( vec.data(), integerConversion< std::ptrdiff_t >( vec.size() ) );
}

/**
 * @brief Return a NumPy view of an ArrayView. This NumPy view may not be resized. If T is const,
 *   the contents may not be modified. The NumPy view will be invalidated if the array is
 *   reallocated.
 * @tparam T type of data that is contained by the array.
 * @tparam NDIM number of dimensions in array
 * @tparam INDEX_TYPE the integer to use for indexing.
 * @tparam BUFFER_TYPE A class that defines how to actually allocate memory for the Array. Must take
 *   one template argument that describes the type of the data being stored (T).
 * @param arr the ArrayView to convert to NumPy.
 */
template< typename T, int NDIM, int USD, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
std::enable_if_t< std::is_arithmetic< T >::value, PyObject * >
create( ArrayView<T, NDIM, USD, INDEX_TYPE, BUFFER_TYPE > const & arr, bool const modify )
{
    arr.move( MemorySpace::CPU, modify );
    return createNumPyArray( arr.data(), modify, NDIM, arr.dims(), arr.strides() );
}

/**
 * @brief create and return a Python list of strings from an ArrayView of std::strings.
 *   the Python strings will be copies.
 * @param arr the ArrayView to convert.
 * @param modify has no effect
 */
template< template< typename > class BUFFER_TYPE >
PyObject * create( ArrayView< std::string, 1, 0, std::ptrdiff_t, BUFFER_TYPE > const & arr, bool const modify )
{
    LVARRAY_UNUSED_VARIABLE( modify );
    arr.move( MemorySpace::CPU, false );
    return internal::createPyListOfStrings( arr.data(), integerConversion< std::ptrdiff_t >( arr.size() ) );
}

} // namespace python
} // namespace LvArray
