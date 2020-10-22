.. _pylvarray:

:mod:`pylvarray` --- LvArray in Python
======================================

.. py:module:: pylvarray
   :synopsis: Manipulate LvArray objects in Python

Many of the LvArray classes can be accessed and manipulated from Python. 
However, they cannot be created from Python.

.. warning:: 
	The ``pylvarray`` module provides plenty of opportunites to crash Python. 
	See the Segmentation Faults section below.

Only Python 3 is supported.

Module Constants
----------------

Space Constants
^^^^^^^^^^^^^^^

The following constants are used to set the space in which an LvArray object lives.
The object ``pylvarray.GPU`` will only be defined if it is a 
valid space for the current system.

.. py:data:: pylvarray.CPU
.. py:data:: pylvarray.GPU

Permissions Constants
^^^^^^^^^^^^^^^^^^^^^

The following constants are used to set permissions for an array instance. 

.. py:data:: pylvarray.READ_ONLY

	No modification of the underlying data is allowed.

.. py:data:: pylvarray.MODIFIABLE

	Allows Numpy views to be modified, but the object itself cannot be resized
	(or otherwise have its buffer reallocated, such as by inserting new elements).

.. py:data:: pylvarray.RESIZEABLE

	Allows Numpy views to be modified, and the object
	to be resized.

Module Classes
--------------

Array and SortedArray
^^^^^^^^^^^^^^^^^^^^^

.. py:class:: Array

	Represents an LvArray::Array, a multidimensional array.

	.. py:method:: get_single_parameter_resize_index()

	.. py:method:: set_single_parameter_resize_index(dim)

		Set the dimension resized by a call to ``resize()``.

	.. py:method:: resize(new_size)

		Resize the array in the default dimension to ``new_size``.

	.. py:method:: resize_all(new_dims)

		Resize all the dimensions of the array in-place, discarding values.

	.. py:method:: to_numpy()

		Return a Numpy view of the array.

	.. py:method:: set_access_level(new_level)

		Set read/modfiy/resize permissions for the instance.

	.. py:method:: get_access_level()

		Return the read/modfiy/resize permissions for the instance.

.. py:class:: SortedArray

	Represents an LvArray::SortedArray, a one-dimensional sorted array.

	.. py:method:: to_numpy()

		Return a Numpy view of the array.

	.. py:method:: set_access_level(new_level)

		Set read/modfiy/resize permissions for the instance.

	.. py:method:: get_access_level()

		Return the read/modfiy/resize permissions for the instance.

	.. py:method:: insert(values)

		Insert one or more values into the array.
		The object passed in will be converted to a 1D numpy array of the same dtype
		as the underlying instance, raising an exception if the conversion cannot be made safely.

	.. py:method:: remove(values)

		Remove one or more values from the array.
		The object passed in will be converted to a 1D numpy array of the same dtype
		as the underlying instance, raising an exception if the conversion cannot be made safely.

ArrayOfArrays and ArrayOfSets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:class:: ArrayOfArrays

	Represents an LvArray::ArrayOfArrays, a two-dimensional ragged array.

	Supports Python's `sequence protocol
	<https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence>`_,
	with the addition of deleting subarrays with ``del arr[i]`` syntax.
	An array fetched with ``[]`` is returned as a Numpy view.
	The built-in ``len()`` function will return the number of arrays in the instance.
	Iterating over an instance will yield a Numpy view of each array.

	.. py:method:: to_numpy()

		Return a Numpy view of the array.

	.. py:method:: set_access_level(new_level)

		Set read/modfiy/resize permissions for the instance.

	.. py:method:: get_access_level()

		Return the read/modfiy/resize permissions for the instance.

	.. py:method:: insert(index, values)

		Insert a new array consisting of ``values`` at the given index.

	.. py:method:: insert_into(index, subindex, values)

		Insert ``values`` into the subarray given by ``index`` at position ``subindex``.
		``values`` will be converted to a 1D numpy array of the same dtype
		as the underlying instance, raising an exception if the conversion cannot be made safely.

	.. py:method:: erase_from(index, subindex)

		Remove the value at ``subindex`` in the subarray ``index``.


.. py:class:: ArrayOfSets

	Represents an LvArray::ArrayOfSets, a collection of sets.

	Behaves very similarly to the ``ArrayOfArrays``, with differences
	outlined below.

	.. py:method:: to_numpy()

		Return a Numpy view of the array.

	.. py:method:: set_access_level(new_level)

		Set read/modfiy/resize permissions for the instance.

	.. py:method:: get_access_level()

		Return the read/modfiy/resize permissions for the instance.

	.. py:method:: insert(position, capacity=0)

		Insert a new set with a given capacity at ``position``.

	.. py:method:: insert_into(set_index, values)

		Insert values into a specific set.

	.. py:method:: erase_from(set_index, values)

		Remove values from a specific set.

CRSMatrix
^^^^^^^^^

.. py:class:: CRSMatrix

	Represents an LvArray::CRSMatrix, a sparse matrix.

	.. py:method:: to_scipy()

		Return a scipy.sparse.csr_matrix representing the matrix.

	.. py:method:: set_access_level(new_level)

		Set read/modfiy/resize permissions for the instance.

	.. py:method:: get_access_level()

		Return the read/modfiy/resize permissions for the instance.

	.. py:method:: num_rows()

		Return the number of rows in the matrix.

	.. py:method:: num_columns()

		Return the number of columns in the matrix.

	.. py:method:: get_entries(row)

		Return a Numpy array representing the entries in the given row.

	.. py:method:: resize(num_rows, num_cols, initial_row_capacity=0)

	.. py:method:: compress()

		Compress the matrix.

	.. py:method:: insert_nonzeros(row, columns, entries)

	.. py:method:: remove_nonzeros(row, columns)

	.. py:method:: add_to_row(row, columns, values)

Segmentation Faults
-------------------
Improper use of this module and associated programs can easily cause Python to crash. 
There are two main causes of crashes.

Stale Numpy Views
^^^^^^^^^^^^^^^^^
The ``pylvarray`` classes provide various ways to get Numpy views of
their data. However, those views are only valid as long as the 
LvArray object's buffer is not reallocated. The buffer may be reallocated
by invoking methods (the ones that require 
the ``RESIZEABLE`` permission) or by calls into a C++ program with access
to the underlying C++ LvArray object.

.. code:: python

	view = my_array.to_numpy()
	my_array.resize(1000)
	print(view)  # segfault

Destroyed LvArray C++ objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As mentioned earlier, the classes defined in this
module cannot be created in Python; some external C++
program must create an LvArray object in C++, then create a
``pylvarray`` view of it. However, the Python view will only be
valid as long as the underlying LvArray C++ object is kept around. If 
that is destroyed, the Python object will be left holding an invalid
pointer and subsequent attempts to use the Python object will cause 
undefined behavior.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

