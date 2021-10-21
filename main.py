import sys
import numpy as np
import torch
import torch.cuda
import argparse
import os
from modules import *
from dataPrepare import GraphDataset, MyData
from settings import *
from logger import Logger
import pandas as pd
import time
from torch_geometric.data import DataLoader, Data
from utils import *
import ipdb
import random
# torch.set_default_tensor_type(torch.DoubleTensor)

clip_grad = 20.0
decay_epoch = 5
lr_decay = 0.8
max_decay = 5

# ANNEAL_RATE = 0.95
# ANNEAL_RATE =0.00003
ANNEAL_RATE =0.00003

def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='News20')
    parser.add_argument('--model_type', type=str, default='GDGNNMODEL')
    parser.add_argument('--prior_type', type=str, default='Dir2')
    parser.add_argument('--enc_nh', type=int, default=128)
    parser.add_argument('--num_topic', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.90)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--init_mult', type=float, default=1.0)  # multiplier in initialization of decoder weight
    parser.add_argument('--device', default='cpu')  # do not use GPU acceleration
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--taskid', type=int, default=0, help='slurm task id')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--ni', type=int, default=300) # 300
    parser.add_argument('--nw', type=int, default=300)

    parser.add_argument('--fixing', action='store_true', default=True)
    parser.add_argument('--STOPWORD', action='store_true', default=True)
    parser.add_argument('--nwindow', type=int, default=5)
    parser.add_argument('--prior', type=float, default=0.5)
    parser.add_argument('--num_samp', type=int, default=1)
    parser.add_argument('--MIN_TEMP', type=float, default=0.3)
    parser.add_argument('--INITIAL_TEMP', type=float, default=1.0)
    parser.add_argument('--maskrate', type=float, default=0.5)

    parser.add_argument('--wdecay', type=float, default= 1e-4)
    parser.add_argument('--word', action='store_true', default= True)
    parser.add_argument('--variance', type=float, default=0.995)  # default variance in prior normal in ProdLDA



    # load_path =
    args = parser.parse_args()
    # load_str = '_eval' if args.eval else ''
    save_dir = ROOTPATH + "/models/%s/%s_%s/" % (args.dataset, args.dataset, args.model_type)
    opt_str = '_%s_m%.2f_lr%.4f' % (args.optimizer, args.momentum, args.learning_rate)

    seed_set = [783435, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    args.seed = seed_set[args.taskid]

    if args.model_type in ['GDGNNMODEL']:
        model_str = '_%s_ns%d_ench%d_ni%d_nw%d_ngram%d_temp%.2f-%.2f' % \
                    (args.model_type, args.num_samp, args.enc_nh, args.ni, args.nw,args.nwindow, args.INITIAL_TEMP,args.MIN_TEMP)
    else:
        raise ValueError("the specific model type is not supported")

    if args.model_type in [ 'GDGNNMODEL']:
        id_ = '%s_topic%d%s_prior_type%s_%.2f%s_%d_%d_stop%s_fix%s_word%s' % \
              (args.dataset, args.num_topic, model_str, args.prior_type,args.prior,
               opt_str, args.taskid, args.seed, str(args.STOPWORD), str(args.fixing), str(args.word))
    else:
        id_ = '%s_topic%d%s%s_%d_%d_stop%s_fix%s' % \
              (args.dataset, args.num_topic, model_str,
               opt_str, args.taskid, args.seed, str(args.STOPWORD), str(args.fixing))

    save_dir += id_
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    print("save dir", args.save_dir)
    args.save_path = os.path.join(save_dir, 'model.pt')
    # print("save path", args.save_path)

    args.log_path = os.path.join(save_dir, "log.txt")
    # print("log path", args.log_path)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if 'cuda' in args.device:
        args.cuda = True
    else:
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return args


def test(model, test_loader, mode='VAL', verbose=True):
    model.eval()  # switch to testing mode
    num_sent = 0
    val_output = {}
    for batch in test_loader:
        batch = batch.to(device)
        batch_size = batch.y.size(0)
        outputs = model.loss(batch)
        for key in outputs:
            if key not in val_output:
                val_output[key] = 0
            val_output[key] += outputs[key].item() * batch_size
        num_sent += batch_size

    if verbose:
        report_str = ' ,'.join(['{} {:.4f}'.format(key, val_output[key] / num_sent) for key in val_output])
        print('--{} {} '.format(mode, report_str))
    return val_output['loss'] / num_sent

def learn_feature(model,loader):
    model.eval()  # switch to testing mode
    thetas = []
    labels = []
    for batch in loader:
        batch = batch.to(device)
        theta = model.get_doctopic(batch)
        thetas.append(theta)
        labels.append(batch.y)
    thetas = torch.cat(thetas, dim=0).detach()
    labels = torch.cat(labels, dim=0).detach()
    return thetas,labels

def eval_doctopic(model, test_loader):
    thetas, labels = learn_feature(model,test_loader)
    thetas=thetas.cpu().numpy()
    labels = labels.cpu().numpy()

    eval_top_doctopic(thetas, labels)
    # eval_km_doctopic(thetas, labels)


def main(args):
    global dataset, device
    print(args)
    device = torch.device(args.device)
    path = todatapath(args.dataset)
    stop_str = '_stop' if args.STOPWORD else ''
    dataset = GraphDataset(path, ngram=args.nwindow, STOPWORD=args.STOPWORD)
    train_idxs = [i for i in range(len(dataset)) if dataset[i].train == 1]
    train_data = dataset[train_idxs]
    val_idxs = [i for i in range(len(dataset)) if dataset[i].train == -1]
    val_data = dataset[val_idxs]
    test_idxs = [i for i in range(len(dataset)) if dataset[i].train == 0]
    test_data = dataset[test_idxs]
    vocab = dataset.vocab
    args.vocab = vocab
    args.vocab_size = len(vocab)

    print('data: %d samples  avg %.2f words' % (len(dataset), dataset.word_count / len(dataset)))
    print('finish reading datasets, vocab size is %d' % args.vocab_size)
    print('dropped sentences: %d' % dataset.dropped)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False,
                              follow_batch=['x', 'edge_id', 'y'])
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, follow_batch=['x', 'edge_id', 'y'])
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, follow_batch=['x', 'edge_id', 'y'])
    whole_edge = dataset.whole_edge
    whole_edge_w = dataset.whole_edge_w
    whole_edge = torch.tensor(whole_edge, dtype=torch.long, device=device)
    print('edge number: %d' % whole_edge.size(1))

    save_edge(whole_edge.cpu().numpy(), whole_edge_w, vocab.id2word_, args.save_dir + '/whole_edge.csv')

    word_vec = np.load(path + '{}d_words{}.npy'.format(args.nw, dataset.stop_str))
    word_vec = torch.from_numpy(word_vec).float()
    if args.model_type == 'GDGNNMODEL':
        model = GDGNNModel(args, word_vec=word_vec, whole_edge=whole_edge).to(device)
    else:
        assert False, 'Unknown model type {}'.format(args.model_type)

    print('paramteres', sum(param.numel() for param in model.parameters()))
    print('trainable paramteres', sum(param.numel() for param in model.parameters() if param.requires_grad==True))
    if args.eval:
        args.temp = args.MIN_TEMP
        print('begin evaluation')
        if args.load_path != '':
            model.load_state_dict(torch.load(args.load_path, map_location=torch.device(device)))
            print("%s loaded" % args.load_path)
        else:
            model.load_state_dict(torch.load(args.save_path, map_location=torch.device(device)))
            print("%s loaded" % args.save_path)
        model.eval()
        test(model, test_loader, 'TEST')
        torch.cuda.empty_cache()
        if 'TMN' in args.dataset:
            refpath = todatapath('TMN')
            data = pd.read_csv(refpath + '/overall%s.csv' % stop_str, header=0, dtype={'label': int, 'train': int})

        else:
            data = pd.read_csv(path + '/overall%s.csv' % stop_str, header=0, dtype={'label': int, 'train': int})
        # data = data[data['train']>0]
        common_texts = [text for text in data['content'].values]
        beta = model.get_beta().detach().cpu().numpy()
        eval_topic(beta, [vocab.id2word(i) for i in range(args.vocab_size)],
                   common_texts=common_texts)

        if args.dataset in LABELED_DATASETS:
            eval_doctopic(model, test_loader)
        savepath = '/'.join(args.save_path.split('/')[:-1])+'discriminator.pt'
        return

    ALTER_TRAIN = True
    # model.set_embedding(word_vec, fix=False)
    if args.optimizer == 'Adam':
        enc_optimizer = torch.optim.Adam(model.enc_params, args.learning_rate, betas=(args.momentum, 0.999),
                                         weight_decay=args.wdecay)
        dec_optimizer = torch.optim.Adam(model.dec_params, args.learning_rate, betas=(args.momentum, 0.999),
                                         weight_decay=args.wdecay)
    else:
        assert False, 'Unknown optimizer {}'.format(args.optimizer)
    best_loss = 1e4
    iter_ = decay_cnt = 0
    args.iter_ = iter_
    args.temp = args.INITIAL_TEMP
    opt_dict = {"not_improved": 0, "lr": args.learning_rate, "best_loss": 1e4}
    log_niter = len(train_loader) // 5
    start = time.time()
    args.iter_threahold = max(30*len(train_loader), 2000)

    for epoch in range(args.num_epoch):
        num_sents = 0
        output_epoch = {}
        model.train()  # switch to training mode
        for batch in train_loader:
            batch = batch.to(device)
            batch_size = batch.y.size(0)
            outputs = model.loss(batch)
            loss = outputs['loss']
            num_sents += batch_size
            # optimize

            dec_optimizer.zero_grad()
            enc_optimizer.zero_grad()

            loss.backward()  # backprop
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            if ALTER_TRAIN:
                if epoch % 2 == 0:
                    dec_optimizer.step()

                else:
                    enc_optimizer.step()
            else:
                enc_optimizer.step()
                dec_optimizer.step()
            # report
            for key in outputs:
                if key not in output_epoch:
                    output_epoch[key] = 0
                output_epoch[key] += outputs[key].item() * batch_size

            if iter_ % log_niter == 0:
                report_str = ' ,'.join(['{} {:.4f}'.format(key, output_epoch[key] / num_sents) for key in output_epoch])
                print(
                    'Epoch {}, iter {}, {}, time elapsed {:.2f}s'.format(epoch, iter_, report_str, time.time() - start))
            iter_ += 1
            args.iter_ = iter_
            ntither=args.iter_-args.iter_threahold
        
            if  ntither>=0 and ntither % 1000 == 0 and args.temp > args.MIN_TEMP:
                args.temp = max(args.temp * math.exp(-ANNEAL_RATE * ntither), args.MIN_TEMP)
                # args.temp = max(args.temp * ANNEAL_RATE, args.MIN_TEMP)
                best_loss = 1e4
                opt_dict["best_loss"] = best_loss
                opt_dict["not_improved"] = 0
                model.load_state_dict(torch.load(args.save_path))
        if ALTER_TRAIN and epoch%2==0:
            continue

        model.eval()  # switch to testing mode
        with torch.no_grad():
            val_loss = test(model, val_loader, 'VAL')
        print(best_loss,opt_dict["best_loss"], args.temp, ALTER_TRAIN)
        if val_loss < best_loss:
            print('update best loss')
            best_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
        if val_loss > opt_dict["best_loss"]:
            opt_dict["not_improved"] += 1
            if opt_dict["not_improved"] >= decay_epoch and epoch >= 15 and args.temp==args.MIN_TEMP:
                opt_dict["best_loss"] = best_loss
                opt_dict["not_improved"] = 0
                opt_dict["lr"] = opt_dict["lr"] * lr_decay
                model.load_state_dict(torch.load(args.save_path))
                print('new lr: %f' % opt_dict["lr"])
                decay_cnt += 1
                if args.optimizer == 'Adam':
                    enc_optimizer = torch.optim.Adam(model.enc_params, args.learning_rate, betas=(args.momentum, 0.999),
                                                     weight_decay=args.wdecay)
                    dec_optimizer = torch.optim.Adam(model.dec_params, args.learning_rate, betas=(args.momentum, 0.999),
                                                     weight_decay=args.wdecay)
                else:
                    assert False, 'Unknown optimizer {}'.format(args.optimizer)
        else:
            opt_dict["not_improved"] = 0
            opt_dict["best_loss"] = val_loss
        if decay_cnt == max_decay:
            break
        with torch.no_grad():
            test(model, test_loader, 'TEST')
            if epoch % 5 == 0:
                # torch.cuda.empty_cache()
                beta = model.get_beta().detach().cpu().numpy()
                if epoch>0 and (epoch)%50==0:
                    data = pd.read_csv(path + '/overall%s.csv' % stop_str, header=0, dtype={'label': int, 'train': int})
                    # data = data[data['train']>0]
                    common_texts = [text for text in data['content'].values]
                    eval_topic(beta, [vocab.id2word(i) for i in range(args.vocab_size)], common_texts=common_texts)
                    if args.model_type in ['GDGNNMODEL']:
                        beta_edge = model.get_beta_edge(False).detach().cpu().numpy()[:, 1:]  # 第0位为非边的权重
                        print_top_pairwords(beta_edge, edge_index=whole_edge.cpu().numpy(), vocab=vocab.id2word_)
                        for k in range(len(beta_edge)):
                            save_edge(whole_edge.cpu().numpy(), weights=beta_edge[k, :], vocab=vocab.id2word_,
                                      fname=args.save_dir + '/beta_edge_False_%d.csv' % k)
                        beta_edge = model.get_beta_edge(True).detach().cpu().numpy()[:, 1:]
                        print_top_pairwords(beta_edge, edge_index=whole_edge.cpu().numpy(), vocab=vocab.id2word_)
                        for k in range(len(beta_edge)):
                            save_edge(whole_edge.cpu().numpy(), weights=beta_edge[k, :], vocab=vocab.id2word_,
                                      fname=args.save_dir + '/beta_edge_True_%d.csv' % k)
                    if args.model_type in ['GDGNNMODEL5']:
                        W = model.get_W().detach().cpu().numpy()
                        print('W', W)
                else:
                    eval_topic(beta, [vocab.id2word(i) for i in range(args.vocab_size)])


                if args.dataset in LABELED_DATASETS:
                    eval_doctopic(model, test_loader)
                
        model.train()

    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    with torch.no_grad():
        test(model, test_loader, 'TEST')
        torch.cuda.empty_cache()
        if 'TMN' in args.dataset:
            refpath = todatapath('TMN')
            data = pd.read_csv(refpath + '/overall%s.csv' % stop_str, header=0, dtype={'label': int, 'train': int})

        else:
            data = pd.read_csv(path + '/overall%s.csv' % stop_str, header=0, dtype={'label': int, 'train': int})        # data = data[data['train'] > 0]
        common_texts = [text for text in data['content'].values]
        beta = model.get_beta().detach().cpu().numpy()
        eval_topic(beta, [vocab.id2word(i) for i in range(args.vocab_size)],common_texts=common_texts)
        if args.dataset in LABELED_DATASETS:
            eval_doctopic(model, test_loader)
        
        if args.model_type in [ 'GDGNNMODEL']:
            beta_edge = model.get_beta_edge(False).detach().cpu().numpy()[:,1:] # 第0位为非边的权重
            print_top_pairwords(beta_edge, edge_index=whole_edge.cpu().numpy(), vocab=vocab.id2word_)
            for k in range(len(beta_edge)):
                save_edge(whole_edge.cpu().numpy(), weights=beta_edge[k, :], vocab=vocab.id2word_,
                          fname=args.save_dir + '/beta_edge_False_%d.csv' % k)
            beta_edge = model.get_beta_edge(True).detach().cpu().numpy()[:,1:]
            print_top_pairwords(beta_edge, edge_index=whole_edge.cpu().numpy(), vocab=vocab.id2word_)
            for k in range(len(beta_edge)):
                save_edge(whole_edge.cpu().numpy(), weights=beta_edge[k, :], vocab=vocab.id2word_,
                          fname=args.save_dir + '/beta_edge_True_%d.csv' % k)
        if args.model_type in ['GDGNNMODEL']:
            W = model.get_W().detach().cpu().numpy()
            print('W', W)


if __name__ == '__main__':
    args = init_config()
    if not args.eval:
        sys.stdout = Logger(args.log_path)
    main(args)
