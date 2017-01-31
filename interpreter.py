# -*- coding:utf-8 -*-

from util import to_words, Dictionary
from seq2seq import Seq2Seq
from chainer import serializers

# path info

DATA_PATH = './data/pair_corpus.txt'
MODEL_PATH = 'data/hege.model'


def interpreter(data_path, model_path):
    """
    インタープリタモード．
    入力に対して返答を行う関数，"exit"を入力すると終了する．
    :param data_path: the path of corpus you made model learn
    :param model_path: the path of model you made learn
    :return:
    """
    # call dictionary class
    dic = Dictionary(data_path)
    print('Vocabulary Size (number of words) :', len(dic.id2word))
    print('')

    # rebuild seq2seq model
    # model = Seq2Seq(len(dic.id2word), feature_num=128, hidden_num=64)
    model = Seq2Seq(len(dic.id2word), feature_num=128, hidden_num=64, batch_size=1)
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
        #print(input_sentence)
        model.initialize()  # cellの状態を初期化
        sentence = model.generate(input_sentence, sentence_limit=len(input_sentence) + 30,
                                  word2id=dic.word2id, id2word=dic.id2word)
        #print("teacher : ", " ".join(input_vocab[1:len(input_vocab) - 1]))
        print("-> ", sentence)
        print('')


def test_run(data_path, model_path):
    """
    動作テスト用関数
    入力は訓練データを使用する，実際に学習した出力が帰ってくるかを確認
    :return:
    """

    dic = Dictionary(data_path)

    print('Vocabulary Size (number of words) :', len(dic.id2word))
    print('')

    # rebuild seq2seq model
    model = Seq2Seq(len(dic.id2word), feature_num=128, hidden_num=64, batch_size=1)
    serializers.load_hdf5(model_path, model)

    # run an interpreter
    for num, sentence in enumerate(dic.sentences):
        input_vocab = to_words(sentence)
        input_vocab.insert(0, "<start>")
        input_vocab.append("<eos>")

        # id変換
        input_sentence = [dic.word2id[word] for word in input_vocab if not dic.word2id.get(word) is None]

        model.initialize()  # initialize cell
        sentence = model.generate(input_sentence, sentence_limit=len(input_sentence) + 30,
                                  word2id=dic.word2id, id2word=dic.id2word)
        print("teacher : ", " ".join(input_vocab[1:len(input_vocab)-1]))
        print("correct :", "".join(dic.sentences[int(len(dic.sentences)/2) + num]))
        print("-> ", sentence)
        print('')

        if num == 10:
            break


if __name__ == '__main__':
    interpreter(DATA_PATH, MODEL_PATH)
    #test_run(DATA_PATH, MODEL_PATH)
