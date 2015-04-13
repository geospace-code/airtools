[![Health](https://landscape.io/github/scienceopen/python-AIRtools/master/landscape.png)](https://landscape.io/github/scienceopen/python-AIRtools/master)
[![Build Status](https://travis-ci.org/scienceopen/python-AIRtools.svg)](https://travis-ci.org/scienceopen/python-AIRtools)
[![Coverage Status](https://coveralls.io/repos/scienceopen/python-AIRtools/badge.svg?branch=master)](https://coveralls.io/r/scienceopen/python-AIRtools?branch=master)

python-AIRtools
===============

Prereqs:
--------
```
conda install --file requirements.txt
```
or
```
pip install -r requirements.txt
```


optional Installation:
---------------------
```
pip install https://github.com/scienceopen/python-AIRtools/archive/master.zip
```

Port of P.C. Hansen's notable AIRtools Matlab suite of inversion / regularization tools

PC Hansen's ReguTools ports are also here.

Notes:
------
``` picard.py ```: The resulting graph does not exactly match Matlab results. This may originate from that 
with Numpy 1.8.1 the output of numpy.linalg.svd and scipy.linalg.svd do NOT exactly match Matlab.

``` kaczmarz.py ```: Kaczmarz ART 

``` maxent.py ```: Maximum Entropy Regularization

``` rzr.py ```: remove unused or little used rows from tomographic projection matrix.


