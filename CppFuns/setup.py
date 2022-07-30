import os
import numpy
from Cython.Build import cythonize


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    cpp_version = "c++11"
    if os.name == "nt":
        ext_comp_args = ['/']
        ext_link_args = []

        library_dirs = []
        libraries = []
    else:
        ext_comp_args = ['-fopenmp', f'-std={cpp_version}']
        ext_link_args = ['-fopenmp', f'-std={cpp_version}']
        library_dirs = []
        libraries = ["m"]
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

    config = Configuration('CppFuns', parent_package, top_path)
    config.add_extension('Graph_',
                         sources=['Graph_.pyx'],
                         include_dirs = [numpy.get_include()],
                         language="c++",

                         extra_compile_args=ext_comp_args,
                         extra_link_args=ext_link_args,

                         library_dirs=library_dirs,
                         libraries=libraries,

                         define_macros=define_macros,
                         )

    config.add_extension('Keep_order_',
                         sources=['Keep_order_.pyx'],
                         include_dirs=[numpy.get_include()],
                         language="c++",

                         extra_compile_args=ext_comp_args,
                         extra_link_args=ext_link_args,

                         library_dirs=library_dirs,
                         libraries=library_dirs,

                         define_macros=define_macros,
                         )

    config.add_extension('CppFuns_',
                         sources=['CppFuns_.pyx'],
                         include_dirs=[numpy.get_include()],
                         language="c++",

                         extra_compile_args=ext_comp_args,
                         extra_link_args=ext_link_args,

                         library_dirs=library_dirs,
                         libraries=library_dirs,

                         define_macros=define_macros,
                         )

    config.ext_modules = cythonize(config.ext_modules, compiler_directives={'language_level': 3})

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
