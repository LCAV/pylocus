Algorithms 
##########

MDS, lateration and other algorithms, used for localization using distances and/or angles.

Local install
*************
To use this package on your computer, run from this folder level (where setup.py is located):
.. code-block:: bash

  pip install -e . 
  
This installs the package using symbolic links, avoiding the need for a reinstall whenever the source code is changed.

You can then import the package and its submodules as usual, for example::

  from pylocus import algorithms

  algorithm.reconstruct_mds(...)

Install
*******
This package is currently not published and can only be installed via the 
above instructions. 
