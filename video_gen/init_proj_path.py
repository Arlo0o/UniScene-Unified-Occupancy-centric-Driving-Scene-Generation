from __future__ import absolute_import, division, print_function

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(osp.dirname(__file__))
