from setuptools import setup

req = ['nose','numpy','scipy','matplotlib']

setup(name='pyAIRtools',
	  description='Python port of Matlab AIRtools and ReguTools regularization toolbox',
	  url='https://github.com/scienceopen/pyAIRtools',
	  author='Michael Hirsch, Ph.D.',
      packages=['airtools'],
      install_requires=req,
      extras_require={'cvxopt':['cvxopt']},
	  )

