# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
Extract conversation text from Cornell Movie Dialogs Corpus.
Output: pair_corpus.txt (like the following format)

<post> <TAB> <its reply>
<post> <TAB> <its reply>
<post> <TAB> <its reply> ...

"""
import re
import os
import unicodedata
import zipfile
from urllib import request
from nltk import word_tokenize
from util import is_english


def unzip():
    request.urlretrieve(
        'http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip',
        './data/cornell_movie_dialogs_corpus.zip')
    with zipfile.ZipFile('./data/cornell_movie_dialogs_corpus.zip') as zf:
        for name in zf.namelist():
            (dirname, filename) = os.path.split(name)
            if dirname == 'cornell movie-dialogs corpus':
                zf.extract(name, './data/')
    os.rename('./data/cornell movie-dialogs corpus', './data/cornell_movie-dialogs_corpus')
    os.system("nkf -Ew  ./data/cornell_movie-dialogs_corpus/movie_lines.txt > "
              "./data/cornell_movie-dialogs_corpus/movie_lines_utf-8.txt")


def main():
    corpus_dic = {}
    for text in open('./data/cornell_movie-dialogs_corpus/movie_lines_utf-8.txt', 'r', encoding='utf-8'):
        seq_list = text.split(" +++$+++ ")
        if len(seq_list) == 5:
            corpus_dic[seq_list[0]] = seq_list[4].split('\n')[0]

    corpus_conv = []
    for text in open('./data/cornell_movie-dialogs_corpus/movie_conversations.txt', 'r', encoding='utf-8'):
        seq_list = text.split(" +++$+++ ")
        conv_list = seq_list[3].replace("'", '').replace("[", '').replace("]\n", '').split(", ")
        corpus_conv.append((conv_list[0], conv_list[1]))

    # remove html tags
    tag_pattern = r"<[^>]*?>"
    tag = re.compile(tag_pattern)
    for tup in corpus_conv[:]:
        if corpus_dic.get(tup[0], False) and corpus_dic.get(tup[1], False):
            corpus_dic[tup[0]] = tag.sub("", corpus_dic[tup[0]])
            corpus_dic[tup[1]] = tag.sub("", corpus_dic[tup[1]])
        else:
            corpus_conv.remove(tup)    # remove a pair of IDs which is not exist in conversation corpus

    # remove the sentences containing '--' or '...'
    tag_pattern = r"\.\.\.|--"
    tag = re.compile(tag_pattern)
    for tup in corpus_conv[:]:
        m1 = tag.search(corpus_dic[tup[0]])
        m2 = tag.search(corpus_dic[tup[1]])
        if m1 is not None or m2 is not None:
            del corpus_dic[tup[0]]
            del corpus_dic[tup[1]]
            corpus_conv.remove(tup)

    with open('data/pair_corpus.txt', 'w', encoding='utf-8') as f:
        for tup in corpus_conv:
            m1 = tag.search(corpus_dic[tup[0]])
            m2 = tag.search(corpus_dic[tup[1]])
            if m1 is not None or m2 is not None:
                print('dame')
            f.write(corpus_dic[tup[0]] + '\t' + corpus_dic[tup[1]] + '\n')


if __name__ == '__main__':
    if not os.path.exists('./data/cornell_movie-dialogs_corpus/'):
        unzip()
    main()