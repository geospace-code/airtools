# AIRtools

[![Actions Status](https://github.com/geospace-code/airtools/workflows/ci/badge.svg)](https://github.com/geospace-code/airtools/actions)
[![PyPI Download stats](http://pepy.tech/badge/airtools)](http://pepy.tech/project/airtools)



Limited subset of P.C. Hansen and J. S. Jørgensen
[AIRtools](http://www2.compute.dtu.dk/~pcha/AIRtoolsII/)
Matlab suite of inversion / regularization tools, along with some ReguTools functions.
Also includes linear constrained least squares solver using cvxopt in `lsqlin.py`

More function are available in Matlab from
[AIRtools 2](https://github.com/jakobsj/AIRToolsII).

```sh
python -m pip install -e .
```

## Usage

* logmart.py: log-MART
* picard.py: Picard Plot
* kaczmarz.py  Kaczmarz ART
* maxent.py: Maximum Entropy Regularization  (from ReguTools)
* rzr.py: remove unused or little used rows from tomographic projection matrix
* lsqlin.py: linear constrained least squares solver
* matlab/logmart.m:  Implementation of log-MART
* fortran/logmart.f90: log-MART in Fortran

Examples: [tests/test_all.py](./tests/test_all.py)

### Tests

Run a comparison of the Python code with the Matlab code in the [matlab](./matlab) directory by:

```sh
python airtools/tests/test_octave.py
```

which runs the Matlab version via
[Oct2Py](https://blink1073.github.io/oct2py/).
