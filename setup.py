from setuptools import setup 

with open('README.rst','r') as f:
	long_description = f.read()
	
setup(name='pyAIRtools',
      version='0.1',
	  description='Python port of Matlab AIRtools and ReguTools regularization toolbox',
	  long_description=long_description,
	  author='Michael Hirsch',
	  url='https://github.com/scienceopen/pyAIRtools',
	  install_requires=['numpy','scipy'],
      packages=['airtools'],
	  )

