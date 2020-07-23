#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_15_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL LvArray_ARRAY_API

#include "Array.hpp"
#include "MallocBuffer.hpp"
#include "python/numpyArrayView.hpp"

#include <Python.h>
#include <numpy/arrayobject.h>

// global Array of ints
static LvArray::Array< long long, 1, RAJA::PERM_I, std::ptrdiff_t, LvArray::MallocBuffer > array1d_int( 30 );
// global 4D Array of doubles
static LvArray::Array< double, 4, RAJA::PERM_LKIJ, std::ptrdiff_t, LvArray::MallocBuffer > array4d_double( 5, 6, 7, 8 );
// global 2D Array of chars
static LvArray::Array< char, 2, RAJA::PERM_JI, std::ptrdiff_t, LvArray::MallocBuffer > array2d_char( 3, 4 );

/**
 * Fetch the global 1D array and return a numpy view of it.
 */
static PyObject *
get_array1d( PyObject *self, PyObject *args )
{
    LVARRAY_UNUSED_VARIABLE( self );
    LVARRAY_UNUSED_VARIABLE( args );
    return LvArray::python::create( array1d_int.toView(), true );
}

/**
 * Clear and initialize the global 1D Array with values beginning at
 * an integer offset.
 * NOTE: this may invalidate previous views of the array!
 */
static PyObject *
set_array1d( PyObject *self, PyObject *args )
{
    LVARRAY_UNUSED_VARIABLE( self );
    int offset;
    if ( !PyArg_ParseTuple( args, "i", &offset ) )
        return NULL;
    forValuesInSlice( array1d_int.toSlice(), [&offset]( long long & value )
    {
        value = offset++;
    } );
    return get_array1d( nullptr, nullptr );
}

/**
 * Multiply the global Array by a factor. Return None.
 */
static PyObject *
multiply_array1d( PyObject *self, PyObject *args )
{
    LVARRAY_UNUSED_VARIABLE( self );
    int factor;
    if ( !PyArg_ParseTuple( args, "i", &factor ) )
        return NULL;
    for (long long& i : array1d_int)
    {
        i = factor * i;
    }
    Py_RETURN_NONE;
}

/**
 * Fetch the global SortedArray of floats and return a numpy view of it.
 */
static PyObject *
get_array4d( PyObject *self, PyObject *args )
{
    LVARRAY_UNUSED_VARIABLE( self );
    LVARRAY_UNUSED_VARIABLE( args );
    return LvArray::python::create( array4d_double.toView(), true );
}

/**
 * Clear and initialize the global SortedArray of floats with `range(start, stop)`
 * NOTE: this may invalidate previous views of the array!
 */
static PyObject *
set_array4d( PyObject *self, PyObject *args )
{
    LVARRAY_UNUSED_VARIABLE( self );
    int offset;
    if ( !PyArg_ParseTuple( args, "i", &offset ) )
        return NULL;
    forValuesInSlice( array4d_double.toSlice(), [&offset]( double & value )
    {
        value = offset++;
    } );
    return get_array4d( nullptr, nullptr );
}

/**
 * Multiply the global Array by a factor. Return None.
 */
static PyObject *
multiply_array4d( PyObject *self, PyObject *args )
{
    LVARRAY_UNUSED_VARIABLE( self );
    int factor;
    if ( !PyArg_ParseTuple( args, "i", &factor ) )
        return NULL;
    for (double& i : array4d_double)
    {
        i = factor * i;
    }
    Py_RETURN_NONE;
}

/**
 * Fetch the global SortedArray of floats and return a numpy view of it.
 */
static PyObject *
get_array2d( PyObject *self, PyObject *args )
{
    LVARRAY_UNUSED_VARIABLE( self );
    LVARRAY_UNUSED_VARIABLE( args );
    return LvArray::python::create( array2d_char.toView(), true );
}

/**
 * Clear and initialize the global SortedArray of floats with `range(start, stop)`
 * NOTE: this may invalidate previous views of the array!
 */
static PyObject *
set_array2d( PyObject *self, PyObject *args )
{
    LVARRAY_UNUSED_VARIABLE( self );
    int offset;
    if ( !PyArg_ParseTuple( args, "i", &offset ) )
        return NULL;
    forValuesInSlice( array2d_char.toSlice(), [&offset]( char & value )
    {
        value = offset++;
    } );
    return get_array2d( nullptr, nullptr );
}

/**
 * Multiply the global Array by a factor. Return None.
 */
static PyObject *
multiply_array2d( PyObject *self, PyObject *args )
{
    LVARRAY_UNUSED_VARIABLE( self );
    int factor;
    if ( !PyArg_ParseTuple( args, "i", &factor ) )
        return NULL;
    for (char & i : array2d_char)
    {
        i = factor * i;
    }
    Py_RETURN_NONE;
}

/**
 * Array of functions and docstrings to export to Python
 */
static PyMethodDef LvArrayFuncs[] = {
    {"set_array1d",  set_array1d, METH_VARARGS,
     "Return the numpy representation of a the global 1D Array after initializing it `range(start, stop)`."},
    {"get_array1d",  get_array1d, METH_NOARGS,
     "Get the numpy representation of the global 1D Array."},
    {"multiply_array1d",  multiply_array1d, METH_VARARGS,
     "Multiply the contents of the global 1D Array."},
    {"set_array4d",  set_array4d, METH_VARARGS,
     "Return the numpy representation of a the global 4D Array after initializing it `range(start, stop)`."},
    {"get_array4d",  get_array4d, METH_NOARGS,
     "Get the numpy representation of the global 4D Array."},
    {"multiply_array4d",  multiply_array4d, METH_VARARGS,
     "Multiply the contents of the global 4D Array."},
    {"set_array2d",  set_array2d, METH_VARARGS,
     "Return the numpy representation of a the global 2D Array after initializing it `range(start, stop)`."},
    {"get_array2d",  get_array2d, METH_NOARGS,
     "Get the numpy representation of the global 2D Array."},
    {"multiply_array2d",  multiply_array2d, METH_VARARGS,
     "Multiply the contents of the global 2D Array."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


/**
 * Initialize the module object for Python with the exported functions
 */
static struct PyModuleDef _testLvArrayPythonInterfacemodule = {
    PyModuleDef_HEAD_INIT,
    "testPythonArray",   /* name of module */
    "Module for testing numpy views of LvArray::Array objects", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    LvArrayFuncs,
    NULL,
    NULL,
    NULL,
    NULL,
};


PyMODINIT_FUNC
PyInit_testPythonArray(void)
{
    import_array();
    return PyModule_Create(&_testLvArrayPythonInterfacemodule);
}
