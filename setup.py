from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("optim.Distributions", ["models/Distributions.pyx"],
        include_dirs=[np.get_include()]),
    Extension("optim.Inventory", ["models/Inventory.pyx"],
        include_dirs=[np.get_include()]),
    Extension("optim.Phonology", ["models/Phonology.pyx"],
        include_dirs=[np.get_include()]),
    Extension("optim.Lexicon", ["models/Lexicon.pyx"],
        include_dirs=[np.get_include()]),
    Extension("optim.Grammar", ["models/Grammar.pyx"],
        include_dirs=[np.get_include()]),
    Extension("optim.MCMC", ["models/MCMC.pyx"],
        include_dirs=[np.get_include()]),
]


setup(name = "modules",
    ext_modules = cythonize(extensions))
