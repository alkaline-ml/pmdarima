# Author: Virgile Fritsch <virgile.fritsch@inria.fr> (originally written
# for sklearn, adapted for pmdarima)
# License: BSD 3 clause

import numpy as np


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('__check_build', parent_package, top_path)
    config.add_extension('_check_build',
                         sources=['_check_build.pyx'],
                         include_dirs=[np.get_include()])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
