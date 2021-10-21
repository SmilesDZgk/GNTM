import numpy as np


class VocabEntry(object):
    """docstring for Vocab"""

    def __init__(self, word2id=None, withpad=True):
        super(VocabEntry, self).__init__()

        if word2id:
            self.word2id = word2id
            self.unk_id = word2id['<unk>']
        else:
            self.word2id = dict()
            # self.unk_id = 3
            if withpad:
                self.word2id['<pad>'] = 0
            # self.word2id['<s>'] = 1
            # self.word2id['</s>'] = 2
            # self.word2id['<unk>'] = self.unk_id

        self.id2word_ = {v: k for k, v in self.word2id.items()}
        self.freq = []

    def __getitem__(self, word):
        return self.word2id.get(word,0)

    def __contains__(self, word):
        return word in self.word2id

    def __len__(self):
        return len(self.word2id)

    def get(self,word, id=0):
        return self.word2id.get(word, id)

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word_[wid] = word
            return wid

        else:
            return self[word]

    def pop(self):
        v = len(self.word2id) - 1
        k = self.id2word_[v]
        del self.word2id[k]
        del self.id2word_[v]

    def append(self, word):
        if word not in self:
            v = len(self.word2id)
            self.word2id[word] = v
            self.id2word_[v] = word

    def id2word(self, wid):
        return self.id2word_[wid]

    def decode_sentence(self, sentence):
        decoded_sentence = []
        for wid_t in sentence:
            wid = wid_t.item()
            decoded_sentence.append(self.id2word_[wid])
        return decoded_sentence

    # def set_freq(self, num):
    #     self.freq.append(int(num))

    # def get_pow_freq(self, freq=None):
    #     if freq is None:
    #         pow_freq = [f ** 0.75 for f in self.freq]
    #     else:
    #         pow_freq = [f ** 0.75 for f in freq]
    #     self.pow_freq = np.array(pow_freq) / sum(pow_freq)

    @staticmethod
    def from_corpus(fname, withpad=True):
        vocab = VocabEntry(withpad=withpad)
        with open(fname) as fin:
            for line in fin:
                vocab.add(line.strip().split(' ')[0])
                #vocab.set_freq(line.strip().split(' ')[1])
            #vocab.freq = [0] + vocab.freq  ####3
        return vocab
