# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from os import path
from keras.utils.data_utils import get_file

def custom_objects():
    from layers import *
    from model import *
    return locals()

def CoreNLP_path():
    SERVER = 'http://nlp.stanford.edu/software/'
    VERSION = 'stanford-corenlp-full-2017-06-09'

    origin = '{server}{version}.zip'.format(server=SERVER, version=VERSION)
    lib_dir = path.join(path.abspath(path.dirname(__file__)), 'lib')

    get_file('/tmp/stanford-corenlp.zip',
             origin=origin,
             cache_dir=lib_dir,
             cache_subdir='',
             extract=True)

    return path.join(lib_dir, VERSION)


def get_word2vec_zip_path():
    SERVER = 'http://nlp.stanford.edu/data/'
    VERSION = 'glove.840B.300d'

    origin = '{server}{version}.zip'.format(server=SERVER, version=VERSION)
    cache_dir = path.join(path.abspath(path.dirname(__file__)), 'data')

    get_file('/tmp/glove.zip',
             origin=origin,
             cache_dir=cache_dir,
             cache_subdir='',
             extract=True)

    return path.join(cache_dir, VERSION) + '.txt'
