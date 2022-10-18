from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("optim.Distributions", ["models/Distributions.py"],
        include_dirs=[np.get_include()]),
    Extension("optim.Inventory", ["models/Inventory.py"],
        include_dirs=[np.get_include()]),
]


setup(name = "modules",
    ext_modules = cythonize(extensions))
