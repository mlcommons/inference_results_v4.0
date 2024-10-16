# from distutils.core import setup, Extension
from setuptools import Extension
from setuptools import setup
from pathlib import Path
import os

dirname = Path(__file__).parent
#include_directories=os.path.join("include")
#print("!!!!!! {} !!!!!!!".format(include_directories))
module1 = Extension('thread_binder',
                    sources = ['thread_bind.cpp', 'kmp_launcher.cpp'],
                    depends=['kmp_launcher.hpp'],
                    extra_compile_args=['-fopenmp'],
                    extra_link_args=['-lgomp']
                    )

setup (name = 'thread_binder',
       version = '0.11',
       description = 'Core binder for indepdendent threads',
       ext_modules = [module1])