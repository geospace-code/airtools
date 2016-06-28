from setuptools import setup
import subprocess

try:
    subprocess.call(['conda','install','--yes','--file','requirements.txt'])
except Exception as e:
    pass

setup(name='pyAIRtools',
	  description='Python port of Matlab AIRtools and ReguTools regularization toolbox',
	  url='https://github.com/scienceopen/pyAIRtools',
      packages=['airtools'],
      install_requires=[],
	  )

