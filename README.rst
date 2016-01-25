.. image:: https://travis-ci.org/scienceopen/pyAIRtools.svg?branch=master
    :target: https://travis-ci.org/scienceopen/pyAIRtools
.. image:: https://coveralls.io/repos/scienceopen/pyAIRtools/badge.svg?branch=master&service=github 
    :target: https://coveralls.io/github/scienceopen/pyAIRtools?branch=master 
.. image:: https://codeclimate.com/github/scienceopen/pyAIRtools/badges/gpa.svg
   :target: https://codeclimate.com/github/scienceopen/pyAIRtools
   :alt: Code Climate

===============
python-AIRtools
===============

Port of P.C. Hansen's notable AIRtools Matlab suite of inversion / regularization tools

PC Hansen's ReguTools ports are also here.

Also includes linear constrained least squares solver using cvxopt in ``lsqlin.py``


Installation
------------
::

    python setup.py develop


============    ===========
Function        Description
============    ===========
picard.py       

kaczmarz.py     Kaczmarz ART 

maxent.py       Maximum Entropy Regularization

rzr.py          remove unused or little used rows from tomographic projection matrix.

lsqlin.py       linear constrained least squares solver
============    ===========

