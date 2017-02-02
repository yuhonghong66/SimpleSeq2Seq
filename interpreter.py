# -*- coding:utf-8 -*-

import argparse
from util import to_words, Dictionary
from seq2seq import Seq2Seq
from chainer import serializers, cuda

# path info
DATA_PATH = './data/pair_corpus.txt'
MODEL_PATH = 'data/hoge.model'

# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default='-1', type=int, help='GPU ID (negative value indicates CPU)')
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
    dic = Dictionary(data_path)
    print('Vocabulary Size (number of words) :', len(dic.id2word))
    print('')

    # rebuild seq2seq model
    model = Seq2Seq(len(dic.id2word), feature_num=256, hidden_num=256, batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(model_path, model)

    # run conversation system
    print('Conversation system is ready to run, please talk to me!')
    print('( If you want to end a talk, please type "exit". )')
    print('')
    while True:
        sentence = input()
        if sentence == 'exit':
            print('See you again!')
            break

        input_vocab = to_words(sentence)
        input_vocab.insert(0, "<start>")
        input_vocab.append("<eos>")

        # convert word into ID
        input_sentence = [dic.word2id[word] for word in input_vocab if not dic.word2id.get(word) is None]

        model.initialize()          # initialize cell
        sentence = model.generate(input_sentence, sentence_limit=len(input_sentence) + 30,
                                  word2id=dic.word2id, id2word=dic.id2word)
        print("-> ", sentence)
        print('')


def test_run(data_path, model_path):
    """
    Test function.
    Input is training data.
    Output have to be the sentence which is correct data in training phase.
    :return:
    """

    dic = Dictionary(data_path)

    print('Vocabulary Size (number of words) :', len(dic.id2word))
    print('')

    # rebuild seq2seq model
    model = Seq2Seq(len(dic.id2word), feature_num=256, hidden_num=256, batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(model_path, model)

    # run an interpreter
    for num, sentence in enumerate(dic.input_list):
        input_vocab = to_words(sentence)
        input_vocab.insert(0, "<start>")
        input_vocab.append("<eos>")

        # convert word into ID
        input_sentence = [dic.word2id[word] for word in input_vocab if not dic.word2id.get(word) is None]

        model.initialize()  # initialize cell
        sentence = model.generate(input_sentence, sentence_limit=len(input_sentence) + 30,
                                  word2id=dic.word2id, id2word=dic.id2word)
        print("teacher : ", " ".join(input_vocab[1:len(input_vocab)-1]))
        print("correct :", "".join(dic.output_list[num]))
        print("-> ", sentence)
        print('')

        if num == 10:
            break


if __name__ == '__main__':
    interpreter(DATA_PATH, MODEL_PATH)
    #test_run(DATA_PATH, MODEL_PATH)
