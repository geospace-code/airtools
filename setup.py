from setuptools import setup

req = ['nose','numpy','scipy','matplotlib']

setup(name='pyAIRtools',
      packages=['airtools'],
	  description='Python port of Matlab AIRtools and ReguTools regularization toolbox',
	  url='https://github.com/scivision/pyAIRtools',
	  author='Michael Hirsch, Ph.D.',
      install_requires=req,
      extras_require={'cvxopt':['cvxopt']},
	  )

