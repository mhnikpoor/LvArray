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

// source includes
#include "numpyConversion.hpp"
#include "../CRSMatrixView.hpp"

namespace LvArray
{

namespace python
{

namespace internal
{
    PyObject * createCRSMatrix( PyObject * entries, PyObject * columns, PyObject * offsets );
    PyObject * crsMatrixError( const char * message );
} // namespace internal


template< typename T, typename COLTYPE, typename INDEX_TYPE, template< typename > class BUFFER_TYPE >
std::enable_if_t< std::is_arithmetic< T >::value, PyObject * >
create( CRSMatrixView<T, COLTYPE const, INDEX_TYPE const, BUFFER_TYPE > const & crsMat, bool const modify )
{
    crsMat.move( MemorySpace::CPU, modify );
    // check that the matrix is compressed
    bool compressed = true;
    for (INDEX_TYPE row = 0; row < crsMat.numRows(); ++row)
    {
        compressed = compressed && crsMat.numNonZeros( row ) == crsMat.nonZeroCapacity( row );
    }
    if( !compressed ){
        return internal::crsMatrixError( "Uncompressed matrices cannot be exported to Python" );
    }
    // check that the matrix has rows
    if ( crsMat.numRows() == 0 ){
        return internal::crsMatrixError( "0-dimensional matrices cannot be exported to Python" );
    }
    // collect the data to pass to the conversion functions
    T * const entries = crsMat.getEntries( 0 );
    COLTYPE const * const columns = crsMat.getColumns( 0 );
    INDEX_TYPE const * const offsets = crsMat.getOffsets();

    INDEX_TYPE const numNonZeros = offsets[ crsMat.numRows() ]; // size of columns and entries
    INDEX_TYPE offsetSize = crsMat.numRows() + 1;
    INDEX_TYPE strides = 1;
    return internal::createCRSMatrix(
                                     createNumPyArray( entries, modify, 1, &numNonZeros, &strides ),
                                     createNumPyArray( columns, false, 1, &numNonZeros, &strides ),
                                     createNumPyArray( offsets, false, 1, &offsetSize, &strides ));
}

} // namespace python
} // namespace LvArray
