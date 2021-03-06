import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('FeiPub', parent_package, top_path)
    config.add_subpackage('CppFuns')
    config.add_data_dir('MPIR')
    config.add_data_dir('Eigen339')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
