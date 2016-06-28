.. image:: https://travis-ci.org/scienceopen/airtools.svg?branch=master
    :target: https://travis-ci.org/scienceopen/airtools
.. image:: https://coveralls.io/repos/scienceopen/airtools/badge.svg?branch=master&service=github 
    :target: https://coveralls.io/github/scienceopen/airtools?branch=master 
.. image:: https://codeclimate.com/github/scienceopen/airtools/badges/gpa.svg
   :target: https://codeclimate.com/github/scienceopen/airtools
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
    
Examples
------------
See ``tests/test.py``. You can literally just paste the code from each test into your console for the function you're interested in. Would you like to submit a pull request for an inversion example making a cool plot? Please do, it should be straighforward!


============    ===========
Function        Description
============    ===========
picard.py       Picard Plot

kaczmarz.py     Kaczmarz ART 

maxent.py       Maximum Entropy Regularization (from ReguTools)

rzr.py          remove unused or little used rows from tomographic projection matrix.

lsqlin.py       linear constrained least squares solver
============    ===========


Tests
-----
You can run a comparison of the Python code with the Matlab code in the ``matlab/`` directory by::

    ./tests/test_octave.py
    
which runs the Matlab version via `Oct2Py <https://blink1073.github.io/oct2py/>`_.

