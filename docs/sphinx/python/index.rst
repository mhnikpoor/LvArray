LvArray in Python
=================
Many of the LvArray classes can be accessed and manipulated from Python. 
However, they cannot be created from Python.

.. warning:: 
	The `pylvarray` module provides plenty of opportunites to crash Python. 
	See the Segmentation Faults section below.

Module Constants
----------------

The following constants are used to set the space in which an LvArray object lives.
The object ``pylvarray.GPU`` will only be defined if it is a 
valid space for the current system.

.. py:data:: pylvarray.CPU
.. py:data:: pylvarray.GPU

The following constants are used to set permissions for an array instance. 
If an array's permissions are set to ``READ_ONLY``, Numpy views of 
it will be read-only. If permissions are set to ``MODIFIABLE`` or ``READ_ONLY``,
the object cannot be resized (or otherwise have its buffer reallocated).

.. py:data:: pylvarray.READ_ONLY
.. py:data:: pylvarray.MODIFIABLE
.. py:data:: pylvarray.RESIZEABLE

Module Classes
--------------

.. automodule:: pylvarray
   :members:

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
	print(view) # segfault

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

