from setuptools import setup 
import subprocess

try:
    subprocess.run(['conda','install','--yes','--file','requirements.txt'])
except Exception as e:
    pass

with open('README.rst','r') as f:
	long_description = f.read()
	
setup(name='pyAIRtools',
	  description='Python port of Matlab AIRtools and ReguTools regularization toolbox',
	  long_description=long_description,
	  author='Michael Hirsch',
	  url='https://github.com/scienceopen/pyAIRtools',
	  )

