pylocus 
=======

.. image:: https://travis-ci.org/LCAV/pylocus.svg?branch=master
    :target: https://travis-ci.org/LCAV/pylocus

Python Localization Package
---------------------------


MDS, lateration and other algorithms, used for localization using distances and/or angles.

Local install
*************

Since this package is in an early development phase, this is the recommended install method. 
The latest updates will not always be immediately available on pip, they will be bundled
in the next package release. By keeping the local git repository up-to-date and redoing the local install
regularly, you can be sure to be using the latest version. 

To perform a local install on your computer, run from this folder level (where setup.py is located):

.. code-block:: bash

  pip install -e . 
  
This installs the package using symbolic links, avoiding the need for a reinstall whenever the source code is changed.

Install
*******

To install from pip, simply run :

.. code-block:: bash

  pip install pylocus

PyPi link : https://pypi.python.org/pypi/pylocus

Requirements
************

Depending on which parts of the project you are using, you might need to install more heavy requirements such as cvxpy and cxopt. These are not included in the default install and can be installed by running the following line.

.. code-block:: bash

  pip install -r requirements.txt

Documentation
*************

You can import the package and its submodules as usual, for example:

.. code-block:: python

  from pylocus import algorithms

  Xhat = algorithms.reconstruct_mds(...)

or

.. code-block:: python

  from pylocus.algorithms import *

  Xhat = reconstruct_mds(...)


Documentation of all available functionalities can be found on ReadTheDocs: http://pylocus.readthedocs.org/en/latest/
