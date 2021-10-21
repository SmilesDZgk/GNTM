from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
import numpy as np
from sklearn.cluster import KMeans
from settings import *
from modules import *
import requests



def print_top_words(beta, feature_names, n_top_words=20, common_texts=None):
    print('---------------Printing the Topics------------------')
    topic_set = []
    beta_idx = np.argsort(beta,axis=-1)[:,::-1]
    for i in range(len(beta)):
        line = " ".join([feature_names[j]
                         for j in beta_idx[i, :n_top_words]])
        topic_set.append(line.split(' '))
        # topics = identify_topic_in_line(line)
        # print('|'.join(topics))
        print('topic{}\t{}'.format(i, line))
    print('---------------End of Topics------------------')
    if common_texts:
        texts = [text.replace(';', '').split(' ') for text in common_texts]
        for top_n in [5, 10, 15, 20, 25]:
            results = []
            for metric in ['c_v', 'c_npmi', 'c_uci']:
                tclist = metric_coherence_gensim(measure=metric,
                                                 top_n=top_n,
                                                 topic_word_distrib=beta,
                                                 vocab=np.array(feature_names),
                                                 texts=texts,
                                                 return_mean=False)
                results.append(np.mean(tclist))
                K = len(tclist)
                tclist = np.sort(tclist)[::-1]
                tclist =np.cumsum(tclist)/ range(1, K+1)
                delta = int(K/10)
                if delta==0:
                    delta=1
                slist =[]
                for i in range(int(K/delta)):
                    slist.append('{:6f}'.format(tclist[(i+1)*delta-1]))
                print(metric, ' '.join(slist))
            print('top_n {} tc {:.6f} npmi {:.6f} uci {:.6f}'.format(top_n, *results))




def get_topic_diversity(beta, n_top_words):
    num_topics = beta.shape[0]
    list_w = np.zeros((num_topics, n_top_words))
    for k in range(num_topics):
        idx = beta[k, :].argsort()[-n_top_words:][::-1]
        list_w[k, :] = idx
    n_unique = len(np.unique(list_w))
    TD = n_unique / (n_top_words * num_topics)
    print('Topic diveristy is: {} in top {} words'.format(TD, n_top_words))


def eval_topic(beta, feature_names, n_top_words=20, common_texts=None,candidate_wordid=None):
    if candidate_wordid is not None:
        beta = beta[:,candidate_wordid]
        feature_names = [feature_names[i] for i in candidate_wordid]
    print_top_words(beta, feature_names, n_top_words, common_texts)
    for n_word in [5,10,15,20,25]:
        get_topic_diversity(beta, n_top_words=n_word)


def purity(labels, preds):
    # type np.array of int
    assert len(labels) == len(preds)
    N = len(labels)
    NC = np.max(labels)+1
    C_dct = {}
    for i in range(NC):
        idxs = np.where(labels == i)[0]
        C_dct[i] = set(idxs)

    NW = np.max(preds)+1
    W_dct = {}
    for i in range(NW):
        idxs = np.where(preds == i)[0]
        W_dct[i] = set(idxs)

    tmp = np.zeros((NW,NC))
    for k in range(NW):
        w = W_dct[k]
        for j in range(NC):
            c = C_dct[j]
            tmp[k,j] = len(w & c)
    tmp = np.max(tmp, axis=1)
    return np.sum(tmp)*1.0/N

def NMI(labels, preds):
    # type np.array of int
    assert len(labels) == len(preds)
    N = len(labels)
    NC = np.max(labels)+1
    C_dct = {}
    for i in range(NC):
        idxs = np.where(labels == i)[0]
        C_dct[i] = set(idxs)

    NW = np.max(preds)+1
    W_dct = {}
    for i in range(NW):
        idxs = np.where(preds == i)[0]
        W_dct[i] = set(idxs)

    I = 0
    for k in range(NW):
        w = W_dct[k]
        wl = len(w)*1.0
        for j in range(NC):
            c = C_dct[j]
            cl = len(c)*1.0
            wcl = len(w & c)*1.0
            if wcl>0:
                I+=wcl/N*np.log((N*wcl)/(wl*cl))
    HW=0
    for k in range(NW):
        w = W_dct[k]
        wl = len(w) * 1.0
        if wl>0:
            HW += -wl/N * np.log(wl/N)

    HC=0
    for j in range(NC):
        c = C_dct[j]
        cl = len(c)*1.0
        if cl>0:
            HC+=-cl/N * np.log(cl/N)

    return 2*I/(HC+HW)


def print_top_pairwords(beta_edge, edge_index, vocab, n_top_words=20):
    print('---------------Printing top pair words------------------')
    for i in range(len(beta_edge)):
        idxs = beta_edge[i].argsort()[::-1][:n_top_words]
        line = ''
        for id in idxs:
            r, c = edge_index[:, id]
            pair = '(%s,%s)' % (vocab[r], vocab[c])
            line += pair + ' '
        print('topic{}\t{}'.format(i, line))
    print('---------------End of top pair words------------------')


def eval_top_doctopic(thetas, labels):
    preds = np.argmax(thetas,axis=-1)
    p = purity(labels,preds)
    nmi = NMI(labels,preds)
    print('top-purity {:.6}, top-NMI {:.6}'.format(p,nmi))


def eval_km_doctopic(thetas,labels):
    N = np.max(labels)+1
    cluster = KMeans(n_clusters=N)
    preds=cluster.fit_predict(thetas)
    p = purity(labels,preds)
    nmi = NMI(labels,preds)
    print('km-purity {:.6}, km-NMI {:.6}'.format(p,nmi))


def save_edge(edge_index, weights, vocab, fname, candidate_idx=None):
    if candidate_idx is not None:
        cand_dct = {idx: 1 for idx in candidate_idx}
    with open(fname, 'w') as fd:
        fd.write('source,target,weight' + '\n')
        for i in range(edge_index.shape[1]):
            r, c = edge_index[:, i]
            if candidate_idx is not None:
                try:
                    cand_dct[r]
                    cand_dct[c]
                except:
                    continue
            w = weights[i]
            fd.write(','.join([vocab[r], vocab[c]] + ['{:.15f}'.format(w)]) + '\n')


def save_beta(beta, vocab, fname):
    with open(fname, 'w') as fd:
        for i in range(len(beta)):
            fd.write(','.join([vocab[i], '{:.10f}'.format(beta[i])]) + '\n')

def todatapath(name):
    if name == 'News20':
        path = NEWS20_ADDR
    elif name == 'BNC':
        path = BNC_ADDR
    elif name == 'TMN':
        path = TMN_ADDR
    elif name == 'Reuters':
        path = Reuters_ADDR
    else:
        assert False, 'Unknown dataset {}'.format(name)
    return path


if __name__ == '__main__':
    pass
    

