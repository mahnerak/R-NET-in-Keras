# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import subprocess
import numpy as np
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


def get_glove_file_path():
    SERVER = 'http://nlp.stanford.edu/data/'
    VERSION = 'glove.840B.300d'

    origin = '{server}{version}.zip'.format(server=SERVER, version=VERSION)
    cache_dir = path.join(path.abspath(path.dirname(__file__)), 'data')

    fname = '/tmp/glove.zip'
    get_file(fname,
             origin=origin,
             cache_dir=cache_dir,
             cache_subdir='',
             extract=True)

    # Remove unnecessary .zip file and keep only extracted .txt version
    os.remove(fname)
    return path.join(cache_dir, VERSION) + '.txt'


def get_fasttext_model_path(target_path=None):

    # Validate target_path
    if target_path is None:
        target_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'lib')
        filename = 'wiki.en'
    elif not os.path.isdir(target_path):
        # If there is a ready fasttext model, then return it
        if os.path.exists(target_path):
            print(target_path, 'Exists... Returning fasttext model')
            return target_path

        target_path, filename = os.path.split(target_path)
        filename = filename.replace('.bin', '')

    fname = '/tmp/wiki.en.zip'
    get_file(fname,
             origin='https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip',
             cache_dir=target_path,
             cache_subdir='',
             extract=True)

    target_path += filename
    if os.path.exists(target_path + '.bin'):
        # os.remove(fname)
        # os.remove(target_path + '.vec')
        print('OK!!!')
    else:
        raise FileNotFoundError('binary file of fasttext wasnt extracted properly')
    return target_path + '.bin'


class FastText(object):
    def __init__(self, fasttext_lib_directory, fasttext_model_path):
        cmds = [fasttext_lib_directory, 'print-word-vectors', fasttext_model_path]
        self.model = subprocess.Popen(cmds, stdout=subprocess.PIPE, stdin=subprocess.PIPE, env=os.environ.copy())

        # Test the model
        print('\nTesting the model...\nPrediction for apple: ')
        item = 'apple\n'
        item = item.encode('utf-8')
        self.model.stdin.write(item)
        result = self.model.stdout.readline()
        result = result[len(item):]
        result = np.fromstring(result, dtype=np.float32, sep=' ')
        print(result)

    def __getitem__(self, item):
        assert type(item) is str
        item = item.lower()
        if not item.endswith('\n'):
            item += '\n'

        item = item.encode('utf-8')
        self.model.stdin.write(item)
        result = self.model.stdout.readline()  # Read result
        result = result[len(item):]            # Take everything but the initial item
        result = np.fromstring(result, dtype=np.float32, sep=' ')

        return result
