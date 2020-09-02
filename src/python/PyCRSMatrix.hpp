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
 * @file
 */

#pragma once

// Source includes
#include "pythonForwardDeclarations.hpp"
#include "numpyHelpers.hpp"
#include "../CRSMatrix.hpp"
#include "../Macros.hpp"
#include "../limits.hpp"
#include "../output.hpp"

// TPL includes
#include <RAJA/RAJA.hpp>

// System includes
#include <string>
#include <memory>
#include <typeindex>

namespace LvArray
{
namespace python
{
namespace internal
{

/**
 *
 */
class PyCRSMatrixWrapperBase
{
public:

  /**
   *
   */
  PyCRSMatrixWrapperBase( ):
    m_accessLevel( static_cast< int >( LvArray::python::PyModify::READ_ONLY ) )
  {}

  /**
   *
   */
  virtual ~PyCRSMatrixWrapperBase() = default;

  /**
   * @brief Return the access level for the array.
   * @return the access level for the array.
   */
  virtual int getAccessLevel() const
  { return m_accessLevel; }

  /**
   * @brief Set the access level for the array.
   */
  virtual void setAccessLevel( int accessLevel, int memorySpace ) = 0;

  /**
   *
   */
  virtual std::string repr() const = 0;

  /**
   *
   */
  virtual std::type_index columnType() const = 0;

  /**
   *
   */
  virtual std::type_index entryType() const = 0;

  /**
   *
   */
  virtual std::ptrdiff_t numRows() const = 0;

  /**
   *
   */
  virtual std::ptrdiff_t numColumns() const = 0;

  /**
   *
   */
  virtual bool isCompressed() const = 0;

  /**
   *
   */
  virtual PyObject * getColumns( std::ptrdiff_t const row ) const = 0;

  /**
   *
   */
  virtual PyObject * getEntries( std::ptrdiff_t const row ) const = 0;

  /**
   *
   */
  virtual std::array< PyObject *, 3 > getEntriesColumnsAndOffsets() const = 0;

  /**
   *
   */
  virtual void resize( std::ptrdiff_t const numRows,
                       std::ptrdiff_t const numCols,
                       std::ptrdiff_t const initialRowCapacity ) = 0;

  /**
   *
   */
  virtual void compress() = 0;

  /**
   *
   */
  virtual std::ptrdiff_t insertNonZeros( std::ptrdiff_t const row,
                                         void const * const cols,
                                         void const * const entries,
                                         std::ptrdiff_t const numCols ) = 0;

  /**
   *
   */
  virtual std::ptrdiff_t removeNonZeros( std::ptrdiff_t const row,
                                         void const * const cols,
                                         std::ptrdiff_t const numCols ) = 0;
  /**
   *
   */
  virtual void addToRow( std::ptrdiff_t const row,
                         void const * const cols,
                         void const * const vals,
                         std::ptrdiff_t const numCols ) const = 0;

protected:
  /// access level for the array
  int m_accessLevel;
};

/**
 *
 */
template< typename T,
          typename COL_TYPE,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
class PyCRSMatrixWrapper : public PyCRSMatrixWrapperBase
{
public:

  PyCRSMatrixWrapper( CRSMatrix< T, COL_TYPE, INDEX_TYPE, BUFFER_TYPE > & matrix ):
    PyCRSMatrixWrapperBase( ),
    m_matrix( matrix )
  {}

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual ~PyCRSMatrixWrapper() = default;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::string repr() const final override
  { return system::demangleType< CRSMatrix< T, COL_TYPE, INDEX_TYPE, BUFFER_TYPE > >(); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::type_index columnType() const final override
  { return typeid( COL_TYPE ); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::type_index entryType() const final override
  { return typeid( T ); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::ptrdiff_t numRows() const final override
  { return m_matrix.numRows(); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::ptrdiff_t numColumns() const final override
  { return m_matrix.numColumns(); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual bool isCompressed() const final override
  {
    #if defined(USE_OPENMP)
      using EXEC_POLICY = RAJA::omp_parallel_for_exec;
      using REDUCE_POLICY = RAJA::omp_reduce;
    #else
      using EXEC_POLICY = RAJA::loop_exec;
      using REDUCE_POLICY = RAJA::seq_reduce;
    #endif

    RAJA::ReduceSum< REDUCE_POLICY, INDEX_TYPE > notCompressed( 0 );

    RAJA::forall< EXEC_POLICY >( RAJA::TypedRangeSegment< INDEX_TYPE >( 0, m_matrix.numRows() ),
      [notCompressed, this] ( INDEX_TYPE const row )
      {
        notCompressed += m_matrix.numNonZeros( row ) != m_matrix.nonZeroCapacity( row );
      }
    );

    return !notCompressed.get();
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual PyObject * getColumns( std::ptrdiff_t const row ) const final override
  {
    INDEX_TYPE const convertedRow = integerConversion< INDEX_TYPE >( row );
    COL_TYPE const * const columns = m_matrix.getColumns( convertedRow );
    INDEX_TYPE const nnz = m_matrix.numNonZeros( convertedRow );
    constexpr INDEX_TYPE unitStride = 1;
    return createNumPyArray( columns, false, 1, &nnz, &unitStride );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual PyObject * getEntries( std::ptrdiff_t const row ) const final override
  {
    INDEX_TYPE const convertedRow = integerConversion< INDEX_TYPE >( row );
    T * const entries = m_matrix.getEntries( convertedRow );
    INDEX_TYPE const nnz = m_matrix.numNonZeros( convertedRow );
    constexpr INDEX_TYPE unitStride = 1;
    return createNumPyArray( entries, getAccessLevel(), 1, &nnz, &unitStride );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::array< PyObject *, 3 > getEntriesColumnsAndOffsets() const final override
  {
    T * const entries = m_matrix.getEntries( 0 );
    COL_TYPE const * const columns = m_matrix.getColumns( 0 );
    INDEX_TYPE const * const offsets = m_matrix.getOffsets();

    INDEX_TYPE const numNonZeros = offsets[ m_matrix.numRows() ]; // size of columns and entries
    INDEX_TYPE offsetSize = m_matrix.numRows() + 1;
    constexpr INDEX_TYPE unitStride = 1;
    return { createNumPyArray( entries, getAccessLevel() >= static_cast< int >( LvArray::python::PyModify::MODIFIABLE ), 1, &numNonZeros, &unitStride ),
             createNumPyArray( columns, false, 1, &numNonZeros, &unitStride ),
             createNumPyArray( offsets, false, 1, &offsetSize, &unitStride ) };
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void setAccessLevel( int accessLevel, int memorySpace ) final override
  {
    LVARRAY_UNUSED_VARIABLE( memorySpace );
    if ( accessLevel >= static_cast< int >( LvArray::python::PyModify::RESIZEABLE ) ){
      // touch
    }
    m_accessLevel = accessLevel;
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void resize( std::ptrdiff_t const numRows,
                       std::ptrdiff_t const numCols,
                       std::ptrdiff_t const initialRowCapacity ) final override
  {
    INDEX_TYPE const convertedNumRows = integerConversion< INDEX_TYPE >( numRows );
    INDEX_TYPE const convertedNumCols = integerConversion< INDEX_TYPE >( numCols );
    INDEX_TYPE const convertedCapacity = integerConversion< INDEX_TYPE >( initialRowCapacity );
    m_matrix.resize( convertedNumRows, convertedNumCols, convertedCapacity ); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void compress() final override
  { m_matrix.compress(); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::ptrdiff_t insertNonZeros( std::ptrdiff_t const row,
                                         void const * const cols,
                                         void const * const entries,
                                         std::ptrdiff_t const numCols ) final override
  {
    INDEX_TYPE const convertedRow = integerConversion< INDEX_TYPE >( row );
    COL_TYPE const * const castedCols = reinterpret_cast< COL_TYPE const * >( cols );
    T const * const castedEntries = reinterpret_cast< T const * >( entries );
    INDEX_TYPE const convertedNumCols = integerConversion< INDEX_TYPE >( numCols );
    return integerConversion< std::ptrdiff_t >( m_matrix.insertNonZeros( convertedRow, castedCols, castedEntries, convertedNumCols ) );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual std::ptrdiff_t removeNonZeros( std::ptrdiff_t const row,
                                         void const * const cols,
                                         std::ptrdiff_t const numCols ) final override
  {
    INDEX_TYPE const convertedRow = integerConversion< INDEX_TYPE >( row );
    COL_TYPE const * const castedCols = reinterpret_cast< COL_TYPE const * >( cols );
    INDEX_TYPE const convertedNumCols = integerConversion< INDEX_TYPE >( numCols );
    return integerConversion< std::ptrdiff_t >( m_matrix.removeNonZeros( convertedRow, castedCols, convertedNumCols ) );
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  virtual void addToRow( std::ptrdiff_t const row,
                         void const * const cols,
                         void const * const vals,
                         std::ptrdiff_t const numCols ) const final override
  {
    INDEX_TYPE const convertedRow = integerConversion< INDEX_TYPE >( row );
    COL_TYPE const * const castedCols = reinterpret_cast< COL_TYPE const * >( cols );
    T const * const castedVals = reinterpret_cast< T const * >( vals );
    INDEX_TYPE const convertedNumCols = integerConversion< INDEX_TYPE >( numCols );
    return m_matrix.template addToRowBinarySearchUnsorted< RAJA::seq_atomic >( convertedRow,
                                                                               castedCols,
                                                                               castedVals,
                                                                               convertedNumCols );
  }

private:
  CRSMatrix< T, COL_TYPE, INDEX_TYPE, BUFFER_TYPE > & m_matrix;
};

/**
 *
 */
PyObject * create( std::unique_ptr< internal::PyCRSMatrixWrapperBase > && matrix );

} // namespace internal

/**
 *
 */
// TODO Support multiple levels of modification.
template< typename T,
          typename COL_TYPE,
          typename INDEX_TYPE,
          template< typename > class BUFFER_TYPE >
PyObject * create( CRSMatrix< T, COL_TYPE, INDEX_TYPE, BUFFER_TYPE > & matrix )
{
  return internal::create( std::make_unique< internal::PyCRSMatrixWrapper< T, COL_TYPE, INDEX_TYPE, BUFFER_TYPE > >( matrix ) );
}

/**
 *
 */
PyTypeObject * getPyCRSMatrixType();

} // namespace python
} // namespace LvArray
