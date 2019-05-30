[![image](https://travis-ci.org/scivision/airtools.svg?branch=master)](https://travis-ci.org/scivision/airtools)
[![image](https://coveralls.io/repos/scivision/airtools/badge.svg?branch=master&service=github)](https://coveralls.io/github/scivision/airtools?branch=master)
[![Maintainability](https://api.codeclimate.com/v1/badges/07d00b91f79c958c073a/maintainability)](https://codeclimate.com/github/scivision/airtools/maintainability)
[![PyPi versions](https://img.shields.io/pypi/pyversions/airtools.svg)](https://pypi.python.org/pypi/airtools)
[![PyPi Download stats](http://pepy.tech/badge/airtools)](http://pepy.tech/project/airtools)

# AIRtools

Limited subset of P.C. Hansen and J. S. Jørgensen
[AIRtools 1.0](http://www2.compute.dtu.dk/~pcha/AIRtoolsII/)
Matlab suite of inversion / regularization tools, along with some ReguTools functions.
Also includes linear constrained least squares solver using cvxopt in `lsqlin.py`

More function are available in Matlab from
[AIRtools 2](https://github.com/jakobsj/AIRToolsII).


## Install

```sh
python -m pip install -e .
```

## Usage

Just paste the code from each test into your console for the function
you're interested in. Would you like to submit a pull request for an
inversion example making a cool plot?

* picard.py: Picard Plot
* kaczmarz.py  Kaczmarz ART
* maxent.py: Maximum Entropy Regularization  (from ReguTools)
* rzr.py: remove unused or little used rows from tomographic projection matrix
* lsqlin.py: linear constrained least squares solver
* matlab/logmart.m:  Implementation of log-MART
* fortran/logmart.f90: log-MART in Fortran


### Examples

[tests/test_all.py](./tests/test_all.py)

### Tests

Run a comparison of the Python code with the Matlab code in the [matlab](./matlab) directory by:

    ./tests/test_octave.py

which runs the Matlab version via
[Oct2Py](https://blink1073.github.io/oct2py/).
