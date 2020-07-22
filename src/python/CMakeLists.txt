set( pylvarray_headers
     python.hpp
     numpyConversion.hpp
     numpyArrayView.hpp
     numpySortedArrayView.hpp
    )

set( pylvarray_sources
     numpyConversion.cpp
    )

message("adding pylvarray library")
blt_add_library( NAME             pylvarray
                 SOURCES          ${pylvarray_sources}
                 HEADERS          ${pylvarray_headers}
                 DEPENDS_ON       lvarray ${lvarray_dependencies} Python3::Python Python3::NumPy
                 )

target_include_directories( pylvarray PUBLIC ${CMAKE_BINARY_DIR}/include )

lvarray_add_code_checks( PREFIX pylvarray )
