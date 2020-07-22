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
 * @file numpySortedArrayView.hpp
 * @brief Contains functions to create a NumPy ndarray from a SortedArrayView.
 */

#pragma once

// Source includes
#include "numpyConversion.hpp"
#include "../SortedArrayView.hpp"

namespace LvArray
{
namespace python
{

/**
 * @brief Return a NumPy view of a SortedArrayView. This NumPy view may not be resized and the 
 *   contents may not be modified. The NumPy view will be invalidated if the array is resized.
 * @tparam T type of data that is contained by the array.
 * @tparam INDEX_TYPE the integer to use for indexing.
 * @tparam BUFFER_TYPE A class that defines how to actually allocate memory for the array. Must take
 *   one template argument that describes the type of the data being stored (T).
 * @param arr the SortedArrayView to convert to NumPy.
 */
template< typename T, typename INDEX_TYPE, template<typename> class BUFFER_TYPE >
std::enable_if_t< std::is_arithmetic< T >::value, PyObject * >
create( SortedArrayView< T const, INDEX_TYPE, BUFFER_TYPE > const & arr, bool const modify )
{
  LVARRAY_UNUSED_VARIABLE( modify );

  arr.move( MemorySpace::CPU );
  INDEX_TYPE const dims = arr.size();
  INDEX_TYPE const strides = 1;
  return createNumPyArray( arr.data(), false, 1, &dims, &strides );
}

} // namespace python
} // namespace LvArray
