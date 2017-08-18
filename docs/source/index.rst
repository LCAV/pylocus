.. pylocus documentation master file, adapted from linvpy on July 19, 2017.
.. You can adapt this file completely to your liking, but it should at least contain the root `toctree` directive.

.. image:: images/EPFL_logo.png
   :align: left
   :width: 20 %
.. image:: images/lcav_logo_.png
   :align: right
   :width: 20 %


|
|
|
|


Welcome to the pylocus documentation !
======================================

pylocus is a Python package designed to solve general localization problems, :math:`min_X L(X)`,
where :math:`X` is a :math:`N` x :math:`d` matrix of coordinates of a point set and
:math:`L(X)` is a cost function adapted to the problem at hand. 

The goal is to find the best location of all or some points in :math:`X`. 

Source code is on GitHub : https://github.com/LCAV/pylocus.

.. Paper of reference: **The regularized tau estimator: A robust and efficient solution 
.. to ill-posed linear inverse problems**, by Martinez-Camara et al. You can find it at: https://arxiv.org/abs/1606.00812

Get it
======

pylocus is available from PyPi and Python 3.6 compatible. If you already have pip installed, simply run : ::

    $ sudo pip install --ignore-installed --upgrade pylocus

.. index::

Module contents
===============


.. rubric:: Point Setups
.. automodule:: pylocus.point_set
.. autosummary::
   :nosignatures:

   PointSet
   AngleSet
   HeterogenousSet

.. rubric:: Distance Algorithms
.. automodule:: pylocus.algorithms
.. autosummary::
   :nosignatures:

   procrustes
   reconstruct_mds
   reconstruct_srls
   reconstruct_dwmds
   reconstruct_acd
   reconstruct_sdp

.. rubric:: Angle and Distance Algorithms
.. autosummary::
   :nosignatures:

   reconstruct_emds

Contribute
==========

If you want to contribute to this project, you may fork our GitHub main repository repository : https://github.com/LCAV/pylocus and submit a pull request from a new branch **type/description**, where **type** can be  **fix**, **feature**, **doc**, or **various**.

Documentation
=============

.. module::
.. automodule:: pylocus.point_set
   :members:
   :show-inheritance:
.. automodule:: pylocus.algorithms
   :members:
   :show-inheritance:
.. automodule:: pylocus.basics
   :members:
   :show-inheritance:
   

Indices and tables
==================

:ref:`genindex`
:ref:`modindex`
:ref:`search`
