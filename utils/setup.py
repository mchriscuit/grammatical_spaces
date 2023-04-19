from Cython.Build import cythonize
from distutils.core import setup
from distutils.extension import Extension

extensions = [Extension("optims.Distributions", ["models/Distributions.py"])]

setup(name="modules", ext_modules=cythonize(extensions))
