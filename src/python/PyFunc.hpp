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
#include "PySortedArray.hpp"
#include "../typeManipulation.hpp"
#include "pythonHelpers.hpp"

namespace LvArray
{
namespace python
{

namespace internal
{

bool err( void );

void callPyFunc( PyObject * func, PyObject ** args, std::ptrdiff_t argc );

} // namespace internal

template< typename ... ARGS >
class PythonFunction
{
public:

  PythonFunction( PyObject * pyfunc ):
    m_function( pyfunc )
  { internal::xincref( pyfunc ); }

  void operator()( ARGS ... args )
  {
    // if an error is already set, don't invoke the callback
    if ( internal::err() ){
      return;
    }
    constexpr std::ptrdiff_t argc = sizeof ... (args);
    PyObject * pyArgs[ argc ];
    std::ptrdiff_t i = 0;
    typeManipulation::forEachArg( [&i, &pyArgs]( auto & arg )
    {
      pyArgs[ i ] = create( arg );
      ++i;
    }, args ... );
    bool successful = true;
    for (i = 0; i < argc; ++i)
    {
      if ( pyArgs[ i ] == nullptr ){
        successful = false;
      }
    }
    if ( !successful ){
      for (i = 0; i < argc; ++i)
      {
        internal::xdecref( pyArgs[ i ] );
      }
      return;
    }
    // callPyFunc steals all of the pyArgs references
    internal::callPyFunc( m_function, pyArgs, argc );
  }
private:
  PyObjectRef<> m_function;
};


} // namespace python
} // namespace LvArray
