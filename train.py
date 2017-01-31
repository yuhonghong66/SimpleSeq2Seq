# -*- coding:utf-8 -*-
"""
Sequence to Sequence model
you can also use batch, gpu
args: --gpu (flg of GPU)
"""

import argparse

import numpy as np
import chainer
from chainer import cuda, optimizers, serializers

from util import to_words, read_short_corpus, make_vocab_dict
from seq2seq import Seq2Seq


# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default='-1', type=int, help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
gpu_device = 1
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(gpu_device).use()
xp = cuda.cupy if args.gpu >= 0 else np


def main():

    ###########################
    #### create dictionary ####
    ###########################

    input_list, output_list = read_short_corpus()
    sentences = input_list + output_list
    id2word, word2id = make_vocab_dict(sentences)

    start_id = len(id2word)
    id2word[start_id] = "<start>"
    word2id["<start>"] = start_id

    end_id = start_id + 1
    id2word[end_id] = "<eos>"
    word2id["<eos>"] = end_id

    padding_num = -1
    id2word[padding_num] = "<pad>"
    word2id["<pad>"] = padding_num

    print('Vocabulary Size (number of words) :', len(id2word))

    ######################
    #### create model ####
    ######################

    n_epoch = 1000
    batchsize = 100
    accum_loss = 0

    model = Seq2Seq(len(id2word), feature_num=128, hidden_num=64, batch_size=batchsize)
    if args.gpu >= 0:
        model.to_gpu()
    optimizer = optimizers.AdaGrad(lr=0.01)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    ##########################
    #### create ID corpus ####
    ##########################

    input_mat = []
    output_mat = []
    max_input_ren = max_output_ren = 0
    for input_text, output_text in zip(input_list, output_list):

        # convert to list
        input_text = to_words(input_text)
        input_text.insert(0, "<start>")
        input_text.append("<eos>")
        output_text = to_words(output_text)
        output_text.append("<eos>")

        # update max sentence length
        max_input_ren = max(max_input_ren, len(input_text))
        max_output_ren = max(max_output_ren, len(output_text))

        input_mat.append([word2id[word] for word in input_text])
        output_mat.append([word2id[word] for word in output_text])

    # padding
    for li in input_mat:
        insert_num = max_input_ren - len(li)
        for _ in range(insert_num):
            li.append(padding_num)
    for li in output_mat:
        insert_num = max_output_ren - len(li)
        for _ in range(insert_num):
            li.append(padding_num)

    print(len(input_mat), len(output_mat))

    # create batch matrix
    input_mat = xp.array(input_mat, dtype=xp.int32).T
    output_mat = xp.array(output_mat, dtype=xp.int32).T
    print(input_mat.shape, output_mat.shape)

    #############################
    #### train seq2seq model ####
    #############################

    for num, epoch in enumerate(range(n_epoch)):
        total_loss = 0
        batch_num = 0
        for i in range(int(len(input_list) / batchsize)):

            # select batch data
            input_batch = input_mat[:, (i * batchsize):(i * batchsize) + batchsize]
            output_batch = output_mat[:, (i * batchsize):(i * batchsize) + batchsize]

            # Encode a sentence
            model.initialize()                     # initialize cell
            model.encode(input_batch, train=True)  # encode (output: hidden Variable)

            # Decode from encoded context
            end_batch = xp.array([word2id["<eos>"] for _ in range(batchsize)])
            first_word = output_batch[0]
            loss, predict_mat = model.decode(end_batch, first_word, train=True)  # <eos>タグ(batchsize分)を初期入力に設定
            next_ids = xp.argmax(predict_mat.data, axis=1)
            accum_loss += loss
            for w_ids in output_batch[1:]:
                loss, predict_mat = model.decode(next_ids, w_ids, train=True)
                next_ids = xp.argmax(predict_mat.data, axis=1)
                accum_loss += loss

            model.cleargrads()                      # 勾配を0に初期化
            accum_loss.backward()                   # 累計損失を使って，逆誤差伝播
            accum_loss.unchain_backward()           # truncate BPTT 誤差逆伝播した変数や関数へのreferenceを削除
            optimizer.update()                      # 最適化ルーチンの実行
            total_loss += float(accum_loss.data)
            batch_num += 1
            print('Epoch: ', num, 'Batch_num', batch_num, 'batch loss: {:.2f}'.format(float(accum_loss.data)))
            accum_loss = 0

        # save model and optimizer
        if (epoch + 1) % 10 == 0:
            print('-----', epoch + 1, ' times -----')
            print('save the model')
            serializers.save_hdf5('data/' + 'min' + str(epoch) + '.model', model)
            print('save the optimizer')
            serializers.save_hdf5('data/' + 'min' + str(epoch) + '.state', optimizer)

        print('Epoch: ', num, 'Total loss: {:.2f}'.format(total_loss))


if __name__ == "__main__":
    main()
