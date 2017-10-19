# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import subprocess
import numpy as np
from os import path
from keras.utils.data_utils import get_file


def custom_objects():
    from layers import VariationalDropout, SharedWeight, WrappedGRU, \
        QuestionAttnGRU, helpers, Argmax, PointerGRU, QuestionPooling, SelfAttnGRU, Slice
    from model import RNet
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
    elif os.path.exists(target_path):
        return target_path
    else:
        raise FileNotFoundError('No fasttext binary file at: ' + target_path)

    fname = '/tmp/wiki.en.zip'
    get_file(fname,
             origin='https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip',
             cache_dir=target_path,
             cache_subdir='',
             extract=True)

    return target_path + '/wiki.en.bin'


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
        self.vector_size = len(result)
        print('Length of word-vector is:', self.vector_size)

    def __getitem__(self, item):
        assert type(item) is str
        initial_item = item
        item = item.lower().replace('/', '').replace('-', '').replace('\\', '').replace('`', '')
        if len(item) == 0 or ' ' in item:
            raise KeyError('Could not process: ' + initial_item)

        if not item.endswith('\n'):
            item += '\n'

        item = item.encode('utf-8')
        self.model.stdin.write(item)
        self.model.stdout.flush()
        result = self.model.stdout.readline()  # Read result
        result = result[len(item):]            # Take everything but the initial item
        result = np.fromstring(result, dtype=np.float32, sep=' ')

        if len(result) != self.vector_size:
            print('Could not process: ' + item)
            raise KeyError('Could not process: ' + initial_item)
        return result
