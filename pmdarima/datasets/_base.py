# -*- coding: utf-8 -*-

from os.path import abspath, dirname, join

# caches anything read from disk to avoid re-reads
_cache = {}


def get_data_path():
    """Get the absolute path to the ``data`` directory"""
    dataset_dir = abspath(dirname(__file__))
    data_dir = join(dataset_dir, 'data')
    return data_dir
