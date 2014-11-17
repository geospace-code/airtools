from distutils.core import setup
import pypkg
setup(name='AIRtools',
      version=pypkg.__version__
      py_modules=['maxent','kaczmarz','picard','rzr'],
      )
