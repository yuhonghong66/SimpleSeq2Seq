# -*- coding:utf-8 -*-

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import argparse
import unicodedata
import pickle
import numpy as np
import  matplotlib.pyplot as plt
from nltk import word_tokenize
from chainer import serializers, cuda
from util import ConvCorpus
from seq2seq import Seq2Seq


# path info
DATA_DIR = './data/corpus/'
MODEL_PATH = 'data/9.model'
LOSS_PATH = './data/loss_data.pkl'

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default='-1', type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--bar', '-b', default='0', type=int, help='whether to show the graph of loss values or not')
args = parser.parse_args()
gpu_device = 0
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(gpu_device).use()


def interpreter(data_path, model_path):
    """
    Run this function, if you want to talk to seq2seq model.
    if you type "exit", finish to talk.
    :param data_path: the path of corpus you made model learn
    :param model_path: the path of model you made learn
    :return:
    """
    # call dictionary class
    corpus = ConvCorpus(file_path=None)
    corpus.load(load_dir=data_path)
    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))
    print('')

    # rebuild seq2seq model
    model = Seq2Seq(len(corpus.dic.token2id), feature_num=300, hidden_num=300, batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(model_path, model)

    # run conversation system
    print('Conversation system is ready to run, please talk to me!')
    print('( If you want to end a talk, please type "exit". )')
    print('')
    while True:
        print('>> ', end='')
        sentence = input()
        if sentence == 'exit':
            print('See you again!')
            break

        input_vocab = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(sentence)]
        input_vocab.reverse()
        input_vocab.insert(0, "<eos>")

        # convert word into ID
        input_sentence = [corpus.dic.token2id[word] for word in input_vocab if not corpus.dic.token2id.get(word) is None]

        model.initialize()          # initialize cell
        sentence = model.generate(input_sentence, sentence_limit=len(input_sentence) + 30,
                                  word2id=corpus.dic.token2id, id2word=corpus.dic)
        print("-> ", sentence)
        print('')


def test_run(data_path, model_path):
    """
    Test function.
    Input is training data.
    Output have to be the sentence which is correct data in training phase.
    :return:
    """

    corpus = ConvCorpus(file_path=None)
    corpus.load(load_dir=data_path)

    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))
    print('')

    # rebuild seq2seq model
    model = Seq2Seq(len(corpus.dic.token2id), feature_num=300, hidden_num=300, batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(model_path, model)

    # run an interpreter
    for num, input_sentence in enumerate(corpus.posts):
        id_sequence = input_sentence.copy()
        input_sentence.reverse()
        input_sentence.insert(0, corpus.dic.token2id["<eos>"])

        model.initialize()  # initialize cell
        sentence = model.generate(input_sentence, sentence_limit=len(input_sentence) + 30,
                                  word2id=corpus.dic.token2id, id2word=corpus.dic)
        print("teacher : ", " ".join([corpus.dic[w_id] for w_id in id_sequence]))
        print("correct :", " ".join([corpus.dic[w_id] for w_id in corpus.cmnts[num]]))
        print("-> ", sentence)
        print('')

        if num == 10:
            break


def show_chart(loss_path):
    """
    Show the graph of Losses for each epochs
    """
    with open(loss_path, mode='rb') as f:
        loss_data = np.array(pickle.load(f))
    row = len(loss_data)
    loop_num = np.array([i + 1 for i in range(row)])
    plt.plot(loop_num, loss_data, label="Loss Value", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Rate of Seq2Seq Model")
    plt.show()


if __name__ == '__main__':
    interpreter(DATA_DIR, MODEL_PATH)
    test_run(DATA_DIR, MODEL_PATH)
    if args.bar:
        show_chart(LOSS_PATH)
