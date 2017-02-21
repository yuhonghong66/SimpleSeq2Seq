# Sequence To Sequence model for Chainer 

This repository is the implementation of simple seq2seq model (not containing attention system).
Seq2seq Model is implemented by [chainer][chainer].
This software aim to learn pairs of sentences 
(e.g. conversation corpus or parallel corpus used in the field of Machine Translation) using seq2seq. 

__Caution__: This model is not used NStepLSTM. 
If you only want to use seq2seq model in chainer, you should use [chainer's formal example][chainer_seq2seq]. 

[chainer]: https://github.com/pfnet/chainer "chainer"
[chainer_seq2seq]: https://github.com/pfnet/chainer/blob/seq2seq/examples/seq2seq/seq2seq.py "chainer_seq2seq"



## Description

These scripts use the corpus; pairs of sentences. 

In this experiment, we train the seq2seq model with movie dialogs 
from the [Cornell Movie-Dialogs Corpus][cornell].
You can get a conversation corpus from the data through the use of `cornell_corpus.py`.
The data is located in `./data/pair_corpus.txt` when you run its script.

If you want to learn own corpus, you should create file has the following format.
    
    <post_sentence><TAB><comment_sentence>

A sentence in the second column is reply to a sentence in the first column 
when you use it as a conversation corpus.
If you use it as a parallel corpus, 
a sentence in the second column is a translation result towards a sentence in the first column.
These sentences have to be separated by TAB.
This file should be located in `./data/pair_corpus.txt` (i.e. the same name). 


[cornell]: https://people.mpi-sws.org/~cristian/Cornell_Movie-Dialogs_Corpus.html "cornell"



***DEMO:***

![Demo](https://github.com/OnizukaLab/SentimentAnalysis/blob/master/image/demo_test.png?raw=true)

## Features

- `seq2seq.py`
  - Sequence-To-Sequence Model implemented by [chainer][chainer].

- `util.py`
  - Module of dictionary and corpus.

- `train.py`
  - Train seq2seq model.

- `interpreter.py`
  - Run the model trained by `train.py`.
  - You can talk to ChatBot or Translator.


## Requirement

- pyenv 
- anaconda3-2.4.0
- chainer (1.5 ~ latest)
- gensim
- nkf


## Usage

1. Run `cornell_corpus.py` to make txt file (named `pair_corpus.txt`).
   
   ~~~
    $ python cornell_corpus.py
   ~~~
   
   If you type this command and run it, 
   python script start to install cornell corpus into own your PC.
   
2. Train the seq2seq model using its text.
   You can train the model by `train.py`.

   ~~~
    $ python train.py
   ~~~
   
   This script does not use GPU by default.
   If you want to use GPU, use the parameter; `--gpu`.
   
   ~~~
    $ python train.py --gpu 1
   ~~~
   
   This script use GPU when you set the GPU flag to 1 like above.

3. Run `interpreter.py` to talk ChatBot trained by you.

## Example

Here, I show the result of learning by using a conversation corpus. 

    $ python interpreter.py
      Vocabulary Size (number of words) : 60000
      Conversation system is ready to run, please talk to me!
      ( If you want to end a talk, please type "exit". )
      
      Hello!
      
      -> hello
      
      Do you want to go to school?
      
      -> No
      
      I'm very hungry.
      
      -> Then I can't help you
      
      exit
      
      -> See you!


## Reference 

Ilya Sutskever, Oriol Vinyals, and Quoc V. Le.
[Sequence to sequence learning with neural networks.][s2s_paper]
In Advances in Neural Information Processing Systems (NIPS 2014).

[s2s_paper]: http://papers.nips.cc/paper/5346-information-based-learning-by-agents-in-unbounded-state-spaces.pdf "s2s_paper"

## Author

[@KChikai](https://github.com/KChikai)

