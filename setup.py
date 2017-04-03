from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "fastexpm",
        ["fastexpm.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='fastexpm',
    ext_modules=cythonize(ext_modules), requires=['qutip']
)
