
req = ['nose','numpy','scipy','matplotlib','cvxopt']

import pip
try:
    import conda.cli
    conda.cli.main('install',*req)
except Exception as e:
    pip.main(['install']+req)

# %%
from setuptools import setup


setup(name='pyAIRtools',
      packages=['airtools'],
	  description='Python port of Matlab AIRtools and ReguTools regularization toolbox',
	  url='https://github.com/scivision/airtools',
	  author='Michael Hirsch, Ph.D.',
	  version = '1.0.0',
      install_requires=req,
	  )

