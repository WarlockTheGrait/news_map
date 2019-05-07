from enum import Enum


class Rnn(Enum):
    SIMPLE = 'Simple'
    GRU = 'GRU'
    LSTM = "LSTM"


class WordModel(Enum):
    WORD2VEC = 'word2vec'
    FASTTEXT = 'new_fastText'
    TWITTER_FASTTEXT = 'twitter_fastText'
    ELMO = 'elmo'
