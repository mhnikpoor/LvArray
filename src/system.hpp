/*
 * Copyright (c) 2020, Lawrence Livermore National Security, LLC and LvArray contributors.
 * All rights reserved.
 * See the LICENSE file for details.
 * SPDX-License-Identifier: (BSD-3-Clause)
 */

/**
 * @file system.hpp
 * @brief Contains functions that interact with the system or runtime environment.
 */

#pragma once

// System includes
#include <vector>
#include <array>
#include <string>
#include <typeinfo>

namespace LvArray
{

/**
 * @brief Contains functions that interact with the system or runtime environment.
 */
namespace system
{

/**
 * @brief Return a demangled stack trace of the last 25 frames.
 * @param lineInfo If @c true then file and line numbers will be added to the
 *   trace if available. This is only supported if @c LVARRAY_ADDR2LINE_EXEC is defined and
 *   normally only works in debug builds.
 * @return A demangled stack trace of the last 25 frames.
 */
std::string stackTrace( bool const lineInfo );

/**
 * @return The demangled name corresponding to the given
 *        mangled name @p name.
 * @param name The mangled name.
 */
std::string demangle( char const * const name );

/**
 * @return A demangled type name corresponding to the type @tparam T.
 * @tparam T The type to demangle.
 */
template< class T >
inline std::string demangleType()
{ return demangle( typeid( T ).name() ); }

/**
 * @return A demangled type name corresponding to the type @tparam T.
 * @tparam T The type to demangle.
 */
template< class T >
inline std::string demangleType( T const & )
{ return demangle( typeid( T ).name() ); }

/**
 * @brief Abort the program, correctly finalizing MPI.
 */
void abort();

/**
 * @brief Print signal information and a stack trace to standard out, optionally aborting.
 * @param sig The signal received.
 * @param exit If true abort execution.
 */
void stackTraceHandler( int const sig, bool const exit );

/**
 * @brief Set the signal handler for common signals.
 * @param handler The signal handler.
 */
void setSignalHandling( void (* handler)( int ) );

/**
 * @brief Rest the signal handling back to the original state.
 */
void resetSignalHandling();

/**
 * @brief A wrapper around @c feenableexcept that work on OSX.
 * @param exceptions The set of floating point exceptions to enable.
 * @return The old exception mask or -1 if there was an error.
 */
int enableFloatingPointExceptions( int const exceptions );

/**
 * @brief A wrapper around @c fedisableexcept that work on OSX.
 * @param exceptions The set of floating point exceptions to disable.
 * @return The old exception mask or -1 if there was an error.
 */
int disableFloatingPointExceptions( int const exceptions );

/**
 * @brief Sets the floating point environment.
 * @details Sets the floating point environment such that FE_DIVBYZERO, FE_OVERFLOW
 *   or FE_INVALID throw exceptions. Denormal numbers are flushed to zero.
 */
void setFPE();

/**
 * @return A string representing @p bytes converted to either
 *   KB, MB, or GB.
 * @param bytes The number of bytes.
 */
std::string calculateSize( size_t const bytes );

} // namespace system
} // namespace LvArray