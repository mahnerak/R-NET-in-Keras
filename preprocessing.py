# from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import codecs

import numpy as np
import json
import os
import argparse

from os import path
from gensim.scripts.glove2word2vec import glove2word2vec
from tqdm import tqdm
from unidecode import unidecode

from utils import CoreNLP_path, get_glove_file_path
from stanford_corenlp_pywrapper.sockwrap import CoreNLP
from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle


def CoreNLP_tokenizer():
    proc = CoreNLP(configdict={'annotators': 'tokenize,ssplit'},
                   corenlp_jars=[path.join(CoreNLP_path(), '*')])

    def tokenize_context(context):
        parsed = proc.parse_doc(context)
        tokens = []
        char_offsets = []
        for sentence in parsed['sentences']:
            tokens += sentence['tokens']
            char_offsets += sentence['char_offsets']

        return tokens, char_offsets

    return tokenize_context


def initialize_fasttext(fasttext_path, fasttext_train_data_path='data/fasttext_train_data.txt'):
    import fasttext

    if fasttext_path.endswith('.bin'):
        fasttext_path = fasttext_path.replace('.bin', '')

    # Train fasttext if it's not present
    if not path.exists(fasttext_path + '.bin'):
        print('No FastText model found at %s', fasttext_path)
        print('Starting FastText model training...')

        # Create data for training if there isn't one
        if not os.path.exists(fasttext_train_data_path):
            print('Preparing data for training...', end='')
            fasttext_data = []

            for sample in tqdm(samples):
                tokens, char_offsets = tokenize(sample['context'])
                fasttext_data.append(' '.join(tokens))
                tokens, char_offsets = tokenize(sample['question'])
                fasttext_data.append(' '.join(tokens))

            with codecs.open(fasttext_train_data_path, 'w', 'utf-8') as f:
                f.write('\n'.join(fasttext_data))
            print('Done')

        print('Started training...')
        model = fasttext.skipgram(fasttext_train_data_path, fasttext_path)
        print('Saving fasttext model to %s ...', fasttext_path, end='')
        print('Done')
    else:
        print('Loading fasttext model...', end='')
        model = fasttext.load_model(fasttext_path + '.bin')
        print('Done')

    def get_word_vector(word):
        try:
            return model[word]
        except KeyError:
            return np.zeros(model.dim)

    return get_word_vector


def word2vec(word2vec_path):
    # Download word2vec data if it's not present yet
    if not path.exists(word2vec_path):
        glove_file_path = get_glove_file_path()
        print('Converting Glove to word2vec...', end='')
        glove2word2vec(glove_file_path, word2vec_path)  # Convert glove to word2vec
        os.remove(glove_file_path)                      # Remove glove file and keep only word2vec
        print('Done')

    print('Reading word2vec data... ', end='')
    model = KeyedVectors.load_word2vec_format(word2vec_path)
    print('Done')

    def get_word_vector(word):
        try:
            return model[word]
        except KeyError:
            return np.zeros(model.vector_size)

    return get_word_vector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word2vec_path', type=str, default='data/word2vec_from_glove_300.vec',
                        help='Word2Vec vectors file path')
    parser.add_argument('--outfile', type=str, default='data/tmp.pkl',
                        help='Desired path to output pickle')
    parser.add_argument('--include_str', action='store_true',
                        help='Include strings: available only if we are obtaining word vectors from GLOVE')
    parser.add_argument('--fasttext_path', type=str,  # default='./fasttext_model',
                        help='Path to fastText model, if there is no such model, it will be trained and saved '
                             'automatically')

    parser.add_argument('data', type=str, help='Data json')
    args = parser.parse_args()

    if not args.outfile.endswith('.pkl'):
        args.outfile += '.pkl'

    print('Reading SQuAD data... ', end='')
    with open(args.data) as fd:
        samples = json.load(fd)
    print('Done!')

    print('Initiating CoreNLP service connection... ', end='')
    tokenize = CoreNLP_tokenizer()
    print('Done!')

    # Determine which model to use fasttext or word2vec (Glove)
    if args.fasttext_path is not None:
        if args.include_str:
            raise ValueError('Include string is available only for word2vec')
        word_vector = initialize_fasttext(fasttext_path=args.fasttext_path)
    else:
        word_vector = word2vec(word2vec_path=args.word2vec_path)

    def parse_sample(context, question, answer_start, answer_end, **kwargs):
        inputs = []
        targets = []

        tokens, char_offsets = tokenize(context)
        try:
            answer_start = [s <= answer_start < e
                            for s, e in char_offsets].index(True)
            targets.append(answer_start)
            answer_end   = [s <= answer_end < e
                            for s, e in char_offsets].index(True)
            targets.append(answer_end)
        except ValueError:
            return None

        tokens = [unidecode(token) for token in tokens]

        context_vecs = [word_vector(token) for token in tokens]
        context_vecs = np.vstack(context_vecs).astype(np.float32)
        inputs.append(context_vecs)

        if args.include_str:
            context_str = [np.fromstring(token, dtype=np.uint8).astype(np.int32)
                           for token in tokens]
            context_str = pad_sequences(context_str, maxlen=25)
            inputs.append(context_str)

        tokens, char_offsets = tokenize(question)
        tokens = [unidecode(token) for token in tokens]

        question_vecs = [word_vector(token) for token in tokens]
        question_vecs = np.vstack(question_vecs).astype(np.float32)
        inputs.append(question_vecs)

        if args.include_str:
            question_str = [np.fromstring(token, dtype=np.uint8).astype(np.int32)
                            for token in tokens]
            question_str = pad_sequences(question_str, maxlen=25)
            inputs.append(question_str)

        return [inputs, targets]

    print('Parsing samples... ', end='')
    samples = [parse_sample(**sample) for sample in tqdm(samples)]
    samples = [sample for sample in samples if sample is not None]
    print('Done!')

    # Transpose
    def transpose(x):
        return map(list, zip(*x))

    data = [transpose(sample) for sample in transpose(samples)]

    print('Writing to file {}... '.format(args.outfile), end='')
    with open(args.outfile, 'wb') as fd:
        pickle.dump(data, fd, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done!')
