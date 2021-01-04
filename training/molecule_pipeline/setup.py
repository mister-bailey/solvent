from setuptools import setup, Extension
import imp
import os.path

p = imp.find_module('numpy')[1]
include_dir = os.path.join(p,'core','include','numpy')
setup_dir = os.path.dirname(os.path.realpath(__file__))

ext = Extension('molecule_pipeline_ext',['molecule_pipeline_ext.cpp','molecule_pipeline_imp.cpp',],
               include_dirs = [include_dir,setup_dir], extra_compile_args=['-std=c++11'])

setup(name='molecule_pipeline', py_modules=['molecule_pipeline'], install_requires=['numpy'], ext_modules=[ext])