pylocus 
=======
Python Localization Package
---------------------------


MDS, lateration and other algorithms, used for localization using distances and/or angles.

Local install
*************
To use this package on your computer, run from this folder level (where setup.py is located):

.. code-block:: bash

  pip install -e . 
  
This installs the package using symbolic links, avoiding the need for a reinstall whenever the source code is changed.

You can then import the package and its submodules as usual, for example:

.. code-block:: python

  from pylocus import algorithms

  Xhat = algorithms.reconstruct_mds(...)

or

.. code-block:: python

  from pylocus.algorithms import *

  Xhat = reconstruct_mds(...)

Install
*******

To install from pip, simply run :
$ sudo pip install pylocus

PyPi link : https://pypi.python.org/pypi/pylocus

Documentation
*************

ReadTheDocs link : http://pylocus.readthedocs.org/en/latest/
