#!/usr/bin/env python
req = ['nose','numpy','scipy',]

from setuptools import setup,find_packages


setup(name='pyAIRtools',
      packages=find_packages(),
	  description='Python port of Matlab AIRtools and ReguTools regularization toolbox',
	  url='https://github.com/scivision/airtools',
	  author='Michael Hirsch, Ph.D.',
	  version = '1.0.0',
      long_description=open('README.rst').read(),
      install_requires=req,
      python_requires='>=2.7',
      extras_require={'plot':['matplotlib'],'cvx':['cvxopt'],
                      'octave':['oct2py']}
	  )

