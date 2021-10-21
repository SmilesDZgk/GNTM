import sys

sys.path.append("../")
from settings import *
import re
import pandas as pd
import numpy as np
import os

stopwords = {
    'max>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax>\'ax': 1,
    'edu': 1,
    'subject': 1,
    'com': 1,
    'r<g': 1,
    '_?w': 1,
    'isc': 1,
    'cx^': 1,
    'usr': 1,
    'uga': 1,
    'sam': 1,
    'mhz': 1,
    'b8f': 1,
    '34u': 1,
    'pl+': 1,
    '1993apr20': 1,
    '1993apr15': 1,
    'xterm': 1,
    'utexas': 1,
    'x11r5': 1,
    'o+r': 1,
    'iastate': 1,
    'udel': 1,
    'uchicago': 1,
    '1993apr21': 1,
    'uxa': 1,
    'argic': 1,
    'optilink': 1,
    'imho': 1,
    'umich': 1,
    'openwindows': 1,
    '1993apr19': 1,
    '1993apr22': 1,
    'unk': 1

}
word_len_threshold = 2

np.random.seed(6)


def load_stopwords(lanague='EN'):
    # url: https://github.com/igorbrigadir/stopwords/blob/master/en/gensim.txt
    if lanague == 'EN':
        stopwords_file = EN_STOP_WORDS
    else:
        stopwords_file = EN_STOP_WORDS
    with open(stopwords_file, mode='r', encoding='utf-8') as reader:
        for line in reader:
            word = line.strip()
            stopwords[word] = 1


def is_num(w):
    try:
        float(w)
        return True
    except:
        False


# load_stopwords(stopwords)
def clean_words(words):
    new_words = []
    for word in words:
        word = word.lower()
        if word in stopwords:
            continue
        word = word.strip('_\[\]\'\".,()*! #@~`\\%^&;:/-+=“”‘’<>{}|?$^&'). \
            replace('isn\'t', '').replace('\'s', '').replace('\'re', ''). \
            replace('\'t', '').replace('\'ll', '').replace('\'m', ''). \
            replace('\'am', '').replace('\'ve', '').replace('\'d', '')
        segs = re.split('[()@.\-/#\\\\"`\[\]=:&<>%\']', word)
        new_word = []
        for s in segs:
            seg = s
            # seg = ps.stem(seg)
            if seg not in stopwords and seg and len(seg) > word_len_threshold and not is_num(seg):
                new_word.append(seg)
        # word = ' '.join(new_word)
        # if word and len(word) > word_len_threshold:
        #     if word not in stopwords:
        #           new_words.append(word)
        new_words.extend(new_word)
    return new_words


def parse_sent(line):
    delim = '[.|?|,|!|;]'
    sents = []
    segs = re.split(delim, line)
    for seg in segs:
        if seg not in delim and len(seg) > 5:
            sents.append(seg)
    return sents


def process_dataset(dataset='AG', STOPWORD=False):
    if STOPWORD:
        load_stopwords()
    if dataset == 'BNC':
        path = BNC_ADDR + '/raw/'
        data1 = open(path + 'train.txt').readlines()
        data1 = pd.DataFrame(data1)
        data1['train'] = 1
        data2 = open(path + 'valid.txt').readlines()
        data2 = pd.DataFrame(data2)
        data2['train'] = -1
        data3 = open(path + 'test.txt').readlines()
        data3 = pd.DataFrame(data3)
        data3['train'] = 0
        data = data1.append(data2)
        data = data.append(data3)
        data.columns = ['content', 'train']
        data['label'] = 1
        data = data[['label', 'content', 'train']]

    elif dataset == 'News20':
        from sklearn.datasets import fetch_20newsgroups
        newsgroups_train = fetch_20newsgroups(subset='train',
                                              remove=('headers', 'quotes','footers'))
        # remove=('headers', 'footers', 'quotes'))
        train_data = [[a, b] for b, a in zip(newsgroups_train.data, newsgroups_train.target)]
        data1 = pd.DataFrame(train_data, columns=['label', 'content'])
        data1['train'] = 1
        newsgroups_test = fetch_20newsgroups(subset='test',
                                             remove=('headers', 'quotes', 'footers'))
        # remove=('headers', 'footers', 'quotes'))
        test_data = [[a, b] for b, a in zip(newsgroups_test.data, newsgroups_test.target)]
        data2 = pd.DataFrame(test_data, columns=['label', 'content'])
        data2['train'] = 0
        N = len(data1)
        ids = np.random.choice(range(N), size=3766, replace=False, p=None)
        data1.iloc[ids, -1] = -1
        data = data1.append(data2)
        data = data.dropna()
        data.to_csv(NEWS20_ADDR + '/raw/data.csv', quoting=1, header=True, index=False)


    elif dataset == 'TMN':
        path = TMN_ADDR + '/raw/'
        data1 = open(path + 'tagmynews.txt').readlines()
        N = int(len(data1)/8)
        ldct={}
        values = []
        for i in range(N):
            content = data1[8*i]
            label = data1[8*i+6]
            if label not in ldct:
                id = len(ldct)
                ldct[label]=id
            l= ldct[label]
            values.append([l, content])
        data = pd.DataFrame(values,columns=[ 'label', 'content'])
        N = len(data)
        SN=9000
        ids = np.random.choice(range(N), size=SN, replace=False, p=None)
        N2 = len(ids)
        ids2 = np.random.choice(range(N2), size=int(SN/3), replace=False, p=None)
        ids2 = np.array(ids)[ids2]
        data['train'] = 1
        data.iloc[ids, -1] = 0
        data.iloc[ids2, -1] = -1

    elif dataset in ['Reuters']:
        path = Reuters_ADDR+'/raw'

        train_path = path+'/training/'
        trains = os.listdir(train_path)
        data1=[]
        for t in trains:
            f = train_path+t
            text = ' '.join([str(l).strip() for l in open(f, 'rb').readlines()])
            data1.append(text)
        data1 = pd.DataFrame(data1, columns=[ 'content'])
        data1['train'] = 1
        N = len(data1)
        ids = np.random.choice(range(N), size=int(N / 8), replace=False, p=None)
        data1.iloc[ids, -1] = -1

        test_path = path + '/test/'
        tests = os.listdir(test_path)
        data2 = []
        for t in tests:
            f = test_path + t
            text = ' '.join([str(l).strip() for l in open(f, 'rb').readlines()])
            data2.append(text)
        data2 = pd.DataFrame(data2, columns=['content'])
        data2['train'] = 0
        data = data1.append(data2)
        data['label']=1
        data=data[['label', 'content', 'train']]

    data = data.reset_index()
    data['idx'] = data.index

    print(data['content'].values[0])
    vocab = {}
    labels = []
    contents = []
    # data = data.iloc[:10, :]
    for i, row in enumerate(data[['label','content']].values):
        if i % 1000 == 0:
            print(i)
        label = row[0]
        content = row[1]
        sents = parse_sent(content)
        new_sents = []
        for sen in sents:
            words = sen.strip().split()
            words = clean_words(words)
            for w in words:
                try:
                    vocab[w.strip()] += 1
                except:
                    vocab[w.strip()] = 1
            new_sents.append(' '.join(words))

        new_sents = ' '.join(new_sents)
        labels.append(int(label))
        contents.append(new_sents)

    data['content'] = contents
    data['label'] = labels
    data = data[data['content'].apply(lambda x: len(x) > 5)]

    return data, vocab


def clean_vocab(path, STOPWORD=False, freq_threshold=5):
    vocab = {}
    if STOPWORD:
        data = pd.read_csv(path + 'overall_stop.csv', header=0, dtype={'label': int})
    else:
        data = pd.read_csv(path + 'overall.csv', header=0, dtype={'label': int})

    for line in data['content'].values:
        sents = line.strip()
        words = sents.strip().split(' ')
        for word in words:
            if word in vocab:
                vocab[word] = vocab[word] + 1
            else:
                vocab[word] = 1
    try:
        del vocab['']
    except:
        pass
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    sorted_vocab = [item for item in sorted_vocab if item[1] > freq_threshold]

    texts = ['{} {}'.format(item[0], item[1]) for item in sorted_vocab]
    if STOPWORD:
        with open(path + '/vocab_stop.txt', mode='w', encoding='utf-8') as writer:
            writer.write('\n'.join(texts))
    else:
        with open(path + '/vocab.txt', mode='w', encoding='utf-8') as writer:
            writer.write('\n'.join(texts))


def select_embedding(path, nw=300, STOPWORD=False):
    word2id = {}
    #word2id['<pad>'] = 0
    words = list(word2id.keys())
    idx = 0
    fname = path + '/vocab_stop.txt' if STOPWORD else path + '/vocab.txt'
    savepath = path + '{}d_words_stop'.format(nw) if STOPWORD else path + '{}d_words'.format(nw)
    with open(fname, 'r') as fd:
        for line in fd:
            w = line.strip().split()[0]
            words.append(w)
            word2id[w] = idx
            idx += 1
    # weights_matrix = np.random.normal(scale=0.6,size=(len(words), nw))
    weights_matrix = np.zeros((len(words), nw))

    words_found = {}

    with open(GLOVE_ADDR + 'glove.840B.{}d.txt'.format(nw), 'rb') as fd:
        for l in fd:
            line = l.decode().split()
            word = line[0]
            # if word in ['pad', 'unk', 'eos']:
            #     import ipdb
            #     ipdb.set_trace()
            try:
                idx = word2id[word]
                weights_matrix[idx] = np.array(line[1:]).astype(np.float)
                words_found[word] = 1
            except:
                continue
    idxlist = []
    for w in words:
        try:
            words_found[w]
            idxlist.append(word2id[w])
        except:
            print(w, len(w))
    # for i in range(4):
    #     if i not in idxlist:
    #         idxlist = [i] + idxlist
    idxlist.sort()
    weights_matrix = weights_matrix[idxlist]
    np.save(savepath, weights_matrix)
    words_dct = {words[k]: 1 for k in idxlist}

    newf = open(fname + '_2', 'w')
    with open(fname, 'r') as fd:
        for line in fd:
            w = line.strip().split()[0]
            try:
                words_dct[w]
                newf.write(line)
            except:
                continue
    print(len(words), len(words_found))

def eos_unk_pad_emd(nw=300):
    EOS_UNK_PAD={}
    with open(GLOVE_ADDR + 'glove.840B.{}d.txt'.format(nw), 'rb') as fd:
        for l in fd:
            line = l.decode().split()
            word = line[0]
            if word in ['pad', 'unk', 'eos']:
                EOS_UNK_PAD[word]=np.array(line[1:]).astype(np.float)

    np.save(GLOVE_ADDR+'EOS_UNK_PAD_{}d.dct'.format(nw), EOS_UNK_PAD)


if __name__ == '__main__':
    data, vocab = process_dataset(dataset='TMN', STOPWORD=True)
    print(len(data))
    print(len(vocab))
    data.to_csv(TMN_ADDR + '/overall_stop.csv', header=True, index=False, quoting=1)
    clean_vocab(TMN_ADDR, freq_threshold=5, STOPWORD=True)
    select_embedding(TMN_ADDR, STOPWORD=True)
    #eos_unk_pad_emd()
