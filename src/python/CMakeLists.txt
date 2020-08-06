set( pylvarray_headers
     python.hpp
     pythonForwardDeclarations.hpp
     pythonHelpers.hpp
     numpyHelpers.hpp
     numpyArrayView.hpp
     PySortedArray.hpp
     PyArray.hpp
     numpyCRSMatrixView.hpp
    )

set( pylvarray_sources
     pythonHelpers.cpp
     numpyHelpers.cpp
     numpyCRSMatrixView.cpp
     numpyArrayView.cpp
     PySortedArray.cpp
     PyArray.cpp
    )

message("adding pylvarray library")
blt_add_library( NAME             pylvarray
                 SOURCES          ${pylvarray_sources}
                 HEADERS          ${pylvarray_headers}
                 DEPENDS_ON       lvarray ${lvarray_dependencies} Python3::Python Python3::NumPy
                 )

target_include_directories( pylvarray PUBLIC ${CMAKE_BINARY_DIR}/include )

lvarray_add_code_checks( PREFIX pylvarray )
