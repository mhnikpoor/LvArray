#include <type_traits>
#include "chai/ManagedArray.hpp"
#include "../../../core/src/math/TensorT/TensorT.h"

namespace internal
{

template<typename>
struct is_tensorT : std::false_type {};

template<>
struct is_tensorT< R1TensorT<3> > : std::true_type{};

template<>
struct is_tensorT< R2TensorT<3> > : std::true_type{};

template<>
struct is_tensorT< R2SymTensorT<3> > : std::true_type{};

template< typename T >
struct is_chaiable
{
  static constexpr bool value = std::is_arithmetic<T>::value ||
                                is_tensorT<T>::value;
};

template < typename T >
class ChaiVector
{
public:

  using value_type = T;
  using size_type = size_t;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = pointer;
  using const_iterator = const_pointer;

  /* Element access. */


  reference at( size_type pos )
  {
    assert( pos >= 0 );
    assert( pos < m_length );
    assert( m_length <= m_array.size() );

    return m_array[ pos ];
  }

  const_reference at( size_type pos ) const
  {
    assert( pos >= 0 );
    assert( pos < m_length );
    assert( m_length <= m_array.size() );

    return m_array[ pos ];
  }

  reference operator[]( size_type pos )
  { return m_array[ pos ]; }

  const_reference operator[]( size_type pos ) const
  { return m_array[ pos ]; }

  reference front()
  { return *begin(); }

  const_reference front() const
  { return *begin(); }

  reference back()
  { return *end(); }

  const_reference back() const
  { return *end(); }

  pointer data()
  { return &m_array[0]; }

  const_pointer data() const
  { return &m_array[0]; }


  /* Iterators. */


  iterator begin()
  { return &m_array[0]; }

  const_iterator begin() const
  { return &m_array[0]; }

  iterator end()
  { return &m_array[m_length]; }

  const_iterator end() const
  { return &m_array[m_length]; }


  /* Capacity */


  bool empty() const
  { return size() == 0; }

  size_type size() const
  { return m_length; }

  void reserve( size_type new_cap )
  {
    if ( new_cap > capacity() )
    {
      realloc( new_cap );
    }
  }

  size_type capacity() const
  { return m_array.size(); }

  void shrink_to_fit()
  { realloc( m_length ); }


  /* Modifiers */

  /* Note does not free the associated memory. To do that call reseize(0); shrink_to_fit(). */
  void clear()
  {
    m_length = 0;
  }


  iterator insert( const_iterator pos, const T& value )
  {
    const size_type index = pos - data();
    reserveForInsert( 1, index );
    m_array[ index ] = value;
  }

  template < typename InputIt >
  iterator insert( iterator pos, InputIt first, InputIt last )
  {
    const size_type index = pos - data();
    const size_type n = last - first;
    reserveForInsert( n, index );

    for( size_type i = 0; i < n; ++i )
    {
      m_array[ index + i ] = *first;
      first++;
    }

    return data() + index;
  }

  iterator erase( iterator pos )
  {
    iterator initial_pos = pos;
    while ( pos != end() )
    {
      *pos = *( pos + 1 );
      pos++;
    }

    m_length--;
    return initial_pos;
  }

  void push_back( const_reference value )
  {
    m_length++;
    if ( m_length > capacity() )
    {
      dynamicRealloc( m_length );
    }

    m_array[ m_length - 1 ] = value;
  }

  void pop_back()
  { erase( end() ); }

  void resize( size_type count )
  {
    m_length = count;
    if ( m_length > capacity() )
    {
      reserve( m_length );
    }
  }

private:

  pointer reserveForInsert( size_type n, size_type pos )
  {
    if ( n == 0 )
    {
      return data() + pos;
    }

    size_type new_length = m_length + n;
    if ( new_length > capacity() )
    {
      dynamicRealloc( new_length );
    }

    pointer const insert_pos = data() + pos;
    pointer cur_pos = data() + m_length - 1;
    for ( ; cur_pos >= insert_pos ; --cur_pos )
    {
      *(cur_pos + n) = *cur_pos;
    }

    m_length = new_length;
    return insert_pos;
  }

  void realloc( size_type capacity )
  {
    if ( m_array.size() == 0 )
    {
      m_array.allocate( capacity );
    }
    else
    {
      m_array.reallocate( capacity );
    }
  }

  void dynamicRealloc( size_type new_length )
  { reserve( 2 * new_length ); }

  chai::ManagedArray<T> m_array;
  size_type m_length;
};

} /* namespace internal */
