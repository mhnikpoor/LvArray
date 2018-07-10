/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory
 *
 * LLNL-CODE-746361
 *
 * All rights reserved. See COPYRIGHT for details.
 *
 * This file is part of the GEOSX Simulation Framework.
 *
 * GEOSX is a free software; you can redistrubute it and/or modify it under
 * the terms of the GNU Lesser General Public Liscense (as published by the
 * Free Software Foundation) version 2.1 dated February 1999.
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#include <type_traits>
#include <iterator>
#include "chai/ManagedArray.hpp"
#include "chai/ArrayManager.hpp"


template < typename T >
class ChaiVector
{
public:

  using value_type = T;
  using size_type = size_t;
  using reference = T&;
  using const_reference = const T&;
  using rvalue_reference = T&&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = pointer;
  using const_iterator = const_pointer;


  /**
   * @brief Default constructor, creates a new empty vector.
   */
  ChaiVector() :
    m_array(),
    m_length( 0 ),
    m_copied( false )
  {}

  /**
   * @brief Creates a new vector of the given length.
   * @param [in] initial_length the initial length of the vector.
   */
  ChaiVector( size_type initial_length ) :
    m_array( initial_length ),
    m_length( 0 ),
    m_copied( false )
  {
    resize( initial_length );
  }

  /**
   * @brief Copy constructor, creates a shallow copy of the given ChaiVector.
   * @param [in] source the ChaiVector to copy.
   * @note The copy is a shallow copy and newly constructed ChaiVector doesn't own the data,
   * as such using push_back or other methods that change the state of the array is dangerous.
   * @note When using multiple memory spaces using the copy constructor can trigger a move.
   */
  ChaiVector( const ChaiVector& source ) :
    m_array( source.m_array ),
    m_length( source.m_length ),
    m_copied( true )
  {}

  /**
   * @brief Move constructor, moves the given ChaiVector into *this.
   * @param [in] source the ChaiVector to move.
   * @note Unlike the copy constructor this can not trigger a memory movement.
   */
  ChaiVector( ChaiVector&& source ) :
    m_array( std::move( source.m_array ) ),
    m_length( source.m_length ),
    m_copied( source.m_copied )
  {
    source.m_length = 0;
  }

  /**
   * @brief Destructor, will destroy the objects and free the memory if it owns the data.
   */
  ~ChaiVector()
  {
    if ( capacity() > 0 && !m_copied )
    {
      clear();
      m_array.free();
    }
  }

  /**
   * @brief Move assignment operator, moves the given ChaiVector into *this.
   * @param [in] source the ChaiVector to move.
   * @return *this.
   */
  ChaiVector& operator=( ChaiVector&& source )
  {
    if ( capacity() > 0 && !m_copied )
    {
      clear();
      m_array.free();
    }

    m_array = std::move( source.m_array );
    m_length = source.m_length;
    m_copied = source.m_copied;
    source.m_length = 0;
    return *this;
  }

  /**
   * @brief Return if this ChaiVector is a copy and therefore does not own its data.
   */
  bool isCopy() const
  { return m_copied; }

  /**
   * @brief Dereference operator for the underlying active pointer.
   * @para [in] pos the index to access.
   * @return a reference to the value at the given index.
   */
  /// @{
  reference operator[]( size_type pos )
  { return m_array[ pos ]; }

  const_reference operator[]( size_type pos ) const
  { return m_array[ pos ]; }
  /// @}

  /**
   * @brief Return a reference to the first value in the array.
   */
  /// @{
  reference front()
  { return m_array[0]; }

  const_reference front() const
  { return m_array[0]; }
  /// @}

  /**
   * @brief Return a reference to the last value in the array.
   */
  /// @{
  reference back()
  { return m_array[ m_length - 1 ]; }

  const_reference back() const
  { return m_array[ m_length  - 1 ]; }
  /// @}

  /**
   * @brief Return a pointer to the data.
   */
  /// @{
  pointer data()
  { return &m_array[0]; }

  const_pointer data() const
  { return &m_array[0]; }
  /// @}

  /**
   * @brief Return a random access iterator to the beginning of the vector.
   */
  /// @{
  iterator begin()
  { return &front(); }

  const_iterator begin() const
  { return &front(); }
  /// @}

  /**
   * @brief Return a random access iterator to one past the end of the vector.
   */
  /// @{
  iterator end()
  { return &back() + 1; }

  const_iterator end() const
  { return &back() + 1; }
  /// @}

  /**
   * @brief Return true iff the vector holds no data.
   */
  bool empty() const
  { return size() == 0; }

  /**
   * @brief Return the number of values held in the vector.
   */
  size_type size() const
  { return m_length; }

  /**
   * @brief Allocate space to hold at least the given number of values.
   * @param [in] new_cap the new capacity.
   */
  void reserve( size_type new_cap )
  {
    if ( new_cap > capacity() )
    {
      realloc( new_cap );
    }
  }

  /**
   * @brief Return the capacity of the vector.
   */
  size_type capacity() const
  { return m_array.size(); }


  /* Modifiers */

  /* Note does not free the associated memory. */
  void clear()
  { resize( 0 ); }

  iterator insert( const_iterator pos, const T& value )
  {
    const size_type index = pos - begin();
    emplace( 1, index );
    m_array[ index ] = value;
    return begin() + index;
  }

  template < typename InputIt >
  iterator insert( const_iterator pos, InputIt first, InputIt last )
  {
    const size_type index = pos - begin();
    const size_type n = std::distance( first, last );
    emplace( n, index );

    for( size_type i = 0; i < n; ++i )
    {
      m_array[ index + i ] = *first;
      first++;
    }

    return begin() + index;
  }

  iterator erase( const_iterator pos )
  {
    const size_type index = pos - begin();
    m_length--;
    for ( size_type i = index; i < m_length; ++i )
    {
      m_array[ i ] = std::move( m_array[ i + 1 ] );
    }

    m_array[ m_length ].~T();

    return begin() + index;
  }

  void push_back( const_reference value )
  {
    m_length++;
    if ( m_length > capacity() )
    {
      dynamicRealloc( m_length );
    }

    new ( &m_array[ m_length - 1] ) T();
    m_array[ m_length - 1 ] = value;
  }

  void pop_back()
  { erase( end() ); }

  void resize( const size_type new_length )
  {
    if ( new_length > capacity() )
    {
      realloc( new_length );
    }

    if ( new_length < m_length )
    {
      for ( size_type i = new_length; i < m_length; ++i )
      {
        m_array[ i ].~T();
      }
    }
    else
    {
      for ( size_type i = m_length; i < new_length; ++i )
      {
        new ( &m_array[ i ] ) T();
      }
    }

    m_length = new_length;
  }


  ChaiVector<T> deep_copy() const
  { 
    return ChaiVector( chai::deepCopy( m_array ), m_length );
  }

private:

  ChaiVector( chai::ManagedArray<T>&& source, size_type length ) :
    m_array( std::move( source ) ),
    m_length( length ),
    m_copied( false )
  {}

  void emplace( size_type n, size_type pos )
  {
    if ( n == 0 )
    {
      return;
    }

    size_type new_length = m_length + n;
    if ( new_length > capacity() )
    {
      dynamicRealloc( new_length );
    }

    for ( size_type i = m_length; i > pos; --i )
    {
      const size_type cur_pos = i - 1;
      new ( &m_array[ cur_pos + n ] ) T( std::move( m_array[ cur_pos ] ) );
    }

    for ( size_type i = 0; i < n; ++i )
    {
      const size_type cur_pos = pos + i;
      new ( &m_array[ cur_pos ] ) T();
    }

    m_length = new_length;
  }

  void realloc( size_type new_capacity )
  {
    const size_type initial_capacity = capacity();
    if ( capacity() == 0 )
    {
      m_array.allocate( new_capacity );
    }
    else
    {
      m_array.reallocate( new_capacity );
    }
  }

  void dynamicRealloc( size_type new_length )
  { reserve( 2 * new_length ); }

  chai::ManagedArray<T> m_array;
  size_type m_length;
  bool m_copied;
};
