# -*- coding:utf-8 -*-
"""
Sample script of Sequence to Sequence model for ChatBot.
This is a train script for seq2seq.py
You can also use Batch and GPU.
args: --gpu (flg of GPU, if you want to use GPU, please write "--gpu 1")
"""

import os
os.environ["CHAINER_TYPE_CHECK"] = "0"

import pickle
import argparse
import numpy as np
import chainer
from chainer import cuda, optimizers, serializers
from util import ConvCorpus
from seq2seq import Seq2Seq


# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default='data/cut_corpus100.txt', type=str, help='Data file directory')
parser.add_argument('--gpu', '-g', default='-1', type=int, help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=1000, type=int, help='number of epochs to learn')
parser.add_argument('--feature_num', '-f', default=1024, type=int, help='dimension of feature layer')
parser.add_argument('--hidden_num', '-hi', default=1024, type=int, help='dimension of hidden layer')
parser.add_argument('--batchsize', '-b', default=100, type=int, help='learning minibatch size')
args = parser.parse_args()

gpu_device = 0
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(gpu_device).use()
xp = cuda.cupy if args.gpu >= 0 else np

data_file = args.data
n_epoch = args.epoch
feature_num = args.feature_num
hidden_num = args.hidden_num
batchsize = args.batchsize


def main():

    ###########################
    #### create dictionary ####
    ###########################

    if os.path.exists('./data/corpus/dictionary.dict'):
        corpus = ConvCorpus(file_path=None)
        corpus.load(load_dir='./data/corpus/')
    else:
        corpus = ConvCorpus(args.data)
        corpus.save(save_dir='./data/corpus/')
    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))

    ######################
    #### create model ####
    ######################

    model = Seq2Seq(len(corpus.dic.token2id), feature_num=feature_num,
                    hidden_num=hidden_num, batch_size=batchsize, gpu_flg=args.gpu)
    if args.gpu >= 0:
        model.to_gpu()
    optimizer = optimizers.Adam(alpha=0.001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))
    # optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    ##########################
    #### create ID corpus ####
    ##########################

    input_mat = []
    output_mat = []
    max_input_ren = max_output_ren = 0

    for input_text, output_text in zip(corpus.posts, corpus.cmnts):

        # convert to list
        input_text.reverse()                               # encode words in a reverse order
        input_text.insert(0, corpus.dic.token2id["<eos>"])
        output_text.append(corpus.dic.token2id["<eos>"])

        # update max sentence length
        max_input_ren = max(max_input_ren, len(input_text))
        max_output_ren = max(max_output_ren, len(output_text))

        input_mat.append(input_text)
        output_mat.append(output_text)

    # padding
    for li in input_mat:
        insert_num = max_input_ren - len(li)
        for _ in range(insert_num):
            li.insert(0, corpus.dic.token2id['<pad>'])
    for li in output_mat:
        insert_num = max_output_ren - len(li)
        for _ in range(insert_num):
            li.append(corpus.dic.token2id['<pad>'])

    # create batch matrix
    input_mat = np.array(input_mat, dtype=np.int32).T
    output_mat = np.array(output_mat, dtype=np.int32).T

    # print(input_mat[:, 0:5])
    # print(output_mat[:, 0:5])
    # for i in range(21):
    #     print(corpus.dic[input_mat[i][0]])

    #############################
    #### train seq2seq model ####
    #############################

    accum_loss = 0
    loss_data = []
    for num, epoch in enumerate(range(n_epoch)):
        total_loss = 0
        batch_num = 0
        perm = np.random.permutation(len(corpus.posts))

        for i in range(0, len(corpus.posts), batchsize):

            # select batch data
            input_batch = input_mat[:, perm[i:i + batchsize]]
            output_batch = output_mat[:, perm[i:i + batchsize]]

            # Encode a sentence
            model.initialize()                     # initialize cell
            model.encode(input_batch, train=True)  # encode (output: hidden Variable)

            # Decode from encoded context
            end_batch = xp.array([corpus.dic.token2id["<start>"] for _ in range(batchsize)])
            first_word = output_batch[0]
            loss, predict_mat = model.decode(end_batch, first_word, train=True)
            next_ids = xp.argmax(predict_mat.data, axis=1)
            accum_loss += loss
            for w_ids in output_batch[1:]:
                loss, predict_mat = model.decode(next_ids, w_ids, train=True)
                next_ids = xp.argmax(predict_mat.data, axis=1)
                accum_loss += loss

            model.cleargrads()                      # initialize all grad to zero
            accum_loss.backward()                   # back propagation
            optimizer.update()
            total_loss += float(accum_loss.data)
            batch_num += 1
            print('Epoch: ', num, 'Batch_num', batch_num, 'batch loss: {:.2f}'.format(float(accum_loss.data)))
            accum_loss = 0

        # save model and optimizer
        if (epoch + 1) % 10 == 0:
            print('-----', epoch + 1, ' times -----')
            print('save the model')
            serializers.save_hdf5('data/' + str(epoch) + '.model', model)
            print('save the optimizer')
            serializers.save_hdf5('data/' + str(epoch) + '.state', optimizer)

        print('Epoch: ', num, 'Total loss: {:.2f}'.format(total_loss))
        loss_data.append(total_loss)

    with open('./data/loss_data.pkl', 'wb') as f:
        pickle.dump(loss_data, f)


if __name__ == "__main__":
    main()