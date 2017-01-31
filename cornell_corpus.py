# -*- coding: utf-8 -*-

"""
Extract convasation text from Cornell Movie Dialogs Corpus.
Output: pair_corpus.txt (like below)

<post> <TAB> <its reply>
<post> <TAB> <its reply>
<post> <TAB> <its reply> ...

"""


def main():
    corpus_dic = {}
    for text in open('./data/cornell_movie-dialogs_corpus/movie_lines_utf-8.txt', 'r', encoding='utf-8'):
        seq_list = text.split(" +++$+++ ")
        if len(seq_list) == 5:
            corpus_dic[seq_list[0]] = seq_list[4].split('\n')[0]

    corpus_conv = set()
    for text in open('./data/cornell_movie-dialogs_corpus/movie_conversations.txt', 'r', encoding='utf-8'):
        seq_list = text.split(" +++$+++ ")
        conv_list = seq_list[3].replace("'", '').replace("[", '').replace("]\n", '').split(", ")
        # conv_list = seq_list[3].translate("[]'").split(', ')
        corpus_conv.add((conv_list[0], conv_list[1]))

    with open('data/pair_corpus.txt', 'w', encoding='utf-8') as f:
        for tup in corpus_conv:
            if corpus_dic.get(tup[0], None) is not None and corpus_dic.get(tup[1], None) is not None:
                f.write(corpus_dic[tup[0]] + '\t' + corpus_dic[tup[1]] + '\n')


if __name__ == '__main__':
    main()