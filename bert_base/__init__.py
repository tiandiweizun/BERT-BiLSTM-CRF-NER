# -*- coding: utf-8 -*-

"""

 @Time    : 2019/1/30 19:09
 @Author  : MaCan (ma_cancan@163.com)
 @File    : __init__.py.py
"""
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')