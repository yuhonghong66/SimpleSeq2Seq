# -*- coding:utf-8 -*-

import argparse
from util import to_words, ConvCorpus
from seq2seq import Seq2Seq
from chainer import serializers, cuda

# path info
DATA_PATH = './data/pair_corpus.txt_temp'
MODEL_PATH = 'data/299.model'

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
    corpus = ConvCorpus(file_path=None)
    corpus.load(load_dir='./data/corpus/')
    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))
    print('')

    # rebuild seq2seq model
    model = Seq2Seq(len(corpus.dic.token2id), feature_num=256, hidden_num=256, batch_size=1, gpu_flg=args.gpu)
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
    corpus.load(load_dir='./data/corpus/')

    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))
    print('')

    # rebuild seq2seq model
    model = Seq2Seq(len(corpus.dic.token2id), feature_num=256, hidden_num=256, batch_size=1, gpu_flg=args.gpu)
    serializers.load_hdf5(model_path, model)

    # run an interpreter
    for num, input_sentence in enumerate(corpus.posts):
        id_sequence = input_sentence
        input_sentence.insert(0, corpus.dic.token2id["<start>"])
        input_sentence.append(corpus.dic.token2id["<eos>"])

        model.initialize()  # initialize cell
        sentence = model.generate(input_sentence, sentence_limit=len(input_sentence) + 30,
                                  word2id=corpus.dic.token2id, id2word=corpus.dic)
        print("teacher : ", " ".join([corpus.dic[w_id] for w_id in id_sequence]))
        print("correct :", " ".join([corpus.dic[w_id] for w_id in corpus.cmnts[num]]))
        print("-> ", sentence)
        print('')

        if num == 30:
            break


if __name__ == '__main__':
    #interpreter(DATA_PATH, MODEL_PATH)
    test_run(DATA_PATH, MODEL_PATH)
