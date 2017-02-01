# Sequence To Sequence model for Chainer 

This repository is the implementation of simple seq2seq model (not containing attention system).
Seq2seq Model is implemented by [chainer][chainer].

__caution__: this model is not used NStepLSTM. 
If you only want to use seq2seq model in chainer, you should use [chainer's formal example][chainer_seq2seq]. 

[chainer]: http://google.com/ "chainer"
[chainer_seq2seq]: http://google.com/ "chainer_seq2seq"



## Description

In this experiment, we train the seq2seq model with movie dialogs 
from the [Cornell Movie-Dialogs Corpus][cornell].

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
  - You can talk to ChatBot.


## Requirement

- pyenv 
- anaconda3-2.4.0
- chainer (1.5 ~ latest)
- gensim 

## Usage

1. Install cornell corpus into own your PC.
2. Run `cornell_corpus.py` to make txt file (named `pair_corpus.txt`).
   
    `$ python cornell_corpus.py`
   
3. Train the seq2seq model using its text.
4. Run `interpreter.py` to talk ChatBot trained by you.

## Example

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

Sutskever, I., Vinyals, O., and Le, Q. .
[Sequence to sequence learning with neural networks.][s2s_paper]
In Advances in Neural Information Processing Systems (NIPS 2014).

[s2s_paper]: http://google.com/ "s2s_paper"

## Author

[@KChikai](https://github.com/KChikai)

