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


Installation
------------
from Terminal::

    git clone --depth 1 https://github.com/sciencopen/pyAIRtools
    conda install --file requirements.txt
    python setup.py develop


============    ===========
Function        Description
============    ===========
picard.py       The resulting graph does not exactly match Matlab results. With Numpy 1.8.1 the output of numpy.linalg.svd and scipy.linalg.svd do NOT exactly match Matlab.

kaczmarz.py     Kaczmarz ART 

maxent.py       Maximum Entropy Regularization

rzr.py          remove unused or little used rows from tomographic projection matrix.
============    ===========

