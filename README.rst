.. image:: https://travis-ci.org/scivision/airtools.svg?branch=master
    :target: https://travis-ci.org/scivision/airtools

.. image:: https://coveralls.io/repos/scivision/airtools/badge.svg?branch=master&service=github 
    :target: https://coveralls.io/github/scivision/airtools?branch=master 

.. image:: https://api.codeclimate.com/v1/badges/07d00b91f79c958c073a/maintainability
   :target: https://codeclimate.com/github/scivision/airtools/maintainability
   :alt: Maintainability

===============
python-AIRtools
===============

Port of P.C. Hansen's notable AIRtools Matlab suite of inversion / regularization tools
PC Hansen's ReguTools ports are also here.

Also includes linear constrained least squares solver using cvxopt in ``lsqlin.py``

.. contents::

Install
-------
::

    python -m pip install -e .
    
Examples
--------
See ``tests/test.py``. 
Just paste the code from each test into your console for the function you're interested in. 
Would you like to submit a pull request for an inversion example making a cool plot? 

================    ===========
Function            Description
================    ===========
picard.py           Picard Plot

kaczmarz.py         Kaczmarz ART 

maxent.py           Maximum Entropy Regularization (from ReguTools)

rzr.py              remove unused or little used rows from tomographic projection matrix.

lsqlin.py           linear constrained least squares solver

matlab/logmart.m    Implementation of log-MART used by Joshua Semeter in several publications
================    ===========


Tests
-----
You can run a comparison of the Python code with the Matlab code in the ``matlab/`` directory by::

    ./tests/test_octave.py
    
which runs the Matlab version via `Oct2Py <https://blink1073.github.io/oct2py/>`_.

