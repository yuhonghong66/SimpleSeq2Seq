import re


def to_words(sentence):
    sentence_list = [re.sub(r"(\w+)(!+|\?+|…+|\.+|,+|~+)", r"\1", word) for word in sentence.split(' ')]
    return sentence_list


def read_short_corpus():
    """
    Read conversation data（post-reply pairs)
    単語数が20単語以下から構成されている文書ペアのみ記録
    :return:　post_sentence_list and reply_sentence_list
    """

    # define sentence and corpus size
    max_length = 20
    max_pair_num = 65100

    # read data from txt file
    corpus_path = './data/pair_corpus.txt'
    input_vocab_list = []
    output_vocab_list = []
    pattern = '(.*?)(\t)(.*?)(\n|\r\n)'
    r = re.compile(pattern)
    for line in open(corpus_path, 'r', encoding='utf-8'):
        m = r.search(line)
        if m is not None:
            if len(m.group(1).split(' ')) <= max_length and len(m.group(3).split(' ')) <= max_length:
                input_vocab_list.append(m.group(1))
                output_vocab_list.append(m.group(3))
        if len(input_vocab_list) == max_pair_num:
            print(max_pair_num, 'of pairs has been collected!')
            break

    return input_vocab_list, output_vocab_list


def make_vocab_dict(sentence_list):
    """
    make dictionary using text data
    :param sentence_list: list of sentences (each type is string)
    :return: id-word dic, word-id dic
    """
    id2word = {}
    word2id = {}
    w_id = 0
    for sentence in sentence_list:
        for word in sentence.split(' '):
            word = re.sub(r"(\w+)(!+|\?+|…+|\.+|,+|~+)", r"\1", word)
            if word not in word2id:
                id2word[w_id] = word
                word2id[word] = w_id
                w_id += 1
    return id2word, word2id


### gensim使わない用試作 ###

class Dictionary:
    def __init__(self, file_path=None):
        self.sentences = []
        self.id2word = {}
        self.word2id = {}

        if file_path is not None:
            self.create_dict(file_path)

    def create_dict(self, file_path):
        corpus_path = file_path
        input_vocab_list = []
        output_vocab_list = []

        # read corpus data (post <TAB> comment)
        pattern = '(.*?)(\t)(.*?)(\n|\r\n)'
        r = re.compile(pattern)
        for line in open(corpus_path, 'r', encoding='utf-8'):
            m = r.search(line)
            if m is not None:
                input_vocab_list.append(m.group(1))
                output_vocab_list.append(m.group(3))
        self.sentences = input_vocab_list + output_vocab_list

        # count words
        w_id = 0
        for sentence in self.sentences:
            for word in sentence.split(' '):
                word = re.sub(r"(\w+)(!+|\?+|…+|\.+|,+|~+)", r"\1", word)
                if word not in self.word2id:
                    self.id2word[w_id] = word
                    self.word2id[word] = w_id
                    w_id += 1

        # insert "<start>" and "<eos>"
        start_id = len(self.id2word)
        self.id2word[start_id] = "<start>"
        self.word2id["<start>"] = start_id
        end_id = start_id + 1
        self.id2word[end_id] = "<eos>"
        self.word2id["<eos>"] = end_id
        padding_num = -1
        self.id2word[padding_num] = "<pad>"
        self.word2id["<pad>"] = padding_num

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(load_path):
        with open(load_path, 'rb') as f:
            return pickle.load(f)
