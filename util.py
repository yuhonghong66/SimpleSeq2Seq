# -*- coding:utf-8 -*-

import re
import pickle
import unicodedata
from gensim import corpora
from nltk import word_tokenize


def to_words(sentence):
    sentence_list = [re.sub(r"(\w+)(!+|\?+|…+|\.+|,+|~+)", r"\1", word) for word in sentence.split(' ')]
    return sentence_list


class ConvCorpus:
    def __init__(self, file_path):
        self.posts = []
        self.cmnts = []
        self.dic = None

        if file_path is not None:
            self._construct_dict(file_path)

    def _construct_dict(self, file_path):
        # define sentence and corpus size
        max_length = 20
        max_pair_num = 65100

        # preprocess
        posts = cmnts = []
        pattern = '(.*?)(\t)(.*?)(\n|\r\n)'
        r = re.compile(pattern)
        for line in open(file_path, 'r', encoding='utf-8'):
            m = r.search(line)
            if m is not None:
                post = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(m.group(1))]
                cmnt = [unicodedata.normalize('NFKC', word.lower()) for word in word_tokenize(m.group(3))]
                if len(post) <= max_length and len(cmnt) <= max_length:
                    posts.append(post)
                    cmnts.append(cmnt)
                if len(posts) == max_pair_num:
                    print(max_pair_num, 'of pairs has been collected!')
                    break

        # construct dictionary
        self.dic = corpora.Dictionary(posts + cmnts, prune_at=None)
        self.dic.filter_extremes(no_below=3, no_above=1.0, keep_n=15000)      # cut the size of dictionary

        # add symbols
        self.dic.token2id['<start>'] = len(self.dic.token2id)
        self.dic.token2id['<eos>'] = len(self.dic.token2id)
        self.dic.token2id['<unk>'] = len(self.dic.token2id)
        self.dic.token2id['<pad>'] = -1

        # make ID corpus
        for post in posts:
            self.posts.append([self.dic.token2id.get(word, self.dic.token2id['<unk>']) for word in post])
        for cmnt in cmnts:
            self.cmnts.append([self.dic.token2id.get(word, self.dic.token2id['<unk>']) for word in cmnt])

    def save(self, save_dir):
        self.dic.save(save_dir + 'dictionary.dict')
        with open(save_dir + 'posts.list', 'wb') as f:
            pickle.dump(self.posts, f)
        with open(save_dir + 'cmnts.list', 'wb') as f:
            pickle.dump(self.cmnts, f)

    def load(self, load_dir):
        self.dic = corpora.Dictionary.load(load_dir + 'dictionary.dict')
        with open(load_dir + 'posts.list', 'rb') as f:
            self.posts = pickle.load(f)
        with open(load_dir + 'cmnts.list', 'rb') as f:
            self.cmnts = pickle.load(f)