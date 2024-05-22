import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
#
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = '1'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))


def load_data(dataset, remove_self_loop=False):
    if dataset == 'ACM':
        return load_acm(remove_self_loop)
    elif dataset == 'DBLP':
        return load_dblp(remove_self_loop)
    elif dataset == 'IMDB':
        return load_imdb(remove_self_loop)
    elif dataset == 'amazon':
        return load_amazon(remove_self_loop)
    elif dataset == 'LastFM':
        return load_lastfm(remove_self_loop)
    elif dataset == 'OGBN-MAG':
        return load_ogb(remove_self_loop)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))


def load_acm(remove_self_loop):
    from openhgnn import GTNDataset
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataset = GTNDataset(name='acm4GTN')
    hg = dataset[0]
    features = hg.ndata['h']
    author_features = features['author']
    paper_features = features['paper']
    subject_features = features['subject']
    labels = hg.ndata['label']['paper']
    train_mask = hg.ndata['train_mask']['paper']
    val_mask = hg.ndata['val_mask']['paper']
    test_mask = hg.ndata['test_mask']['paper']
    num_classes = 3
    label = 'paper'
    etypes = ['author-paper', 'paper-author', 'paper-subject', 'subject-paper']
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    return hg.to(device), etypes, {'paper': paper_features.to(device), 'author': author_features.to(device),
                                   'subject': subject_features.to(device)}, \
           label, labels.to(device), num_classes, \
           train_mask.to(device), val_mask.to(device), test_mask.to(device)


def load_dblp(remove_self_loop):
    from openhgnn import GTNDataset
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataset = GTNDataset(name='dblp4GTN')
    hg = dataset[0]
    features = hg.ndata['h']
    author_features = features['author']
    paper_features = features['paper']
    conference_features = features['conference']
    labels = hg.ndata['label']['author']
    train_mask = hg.ndata['train_mask']['author']
    val_mask = hg.ndata['val_mask']['author']
    test_mask = hg.ndata['test_mask']['author']
    num_classes = 4
    label = 'author'
    etypes = ['author-paper', 'paper-author', 'paper-conference', 'conference-paper']
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    return hg.to(device), etypes, {'paper': paper_features.to(device), 'author': author_features.to(device),
                                   'conference': conference_features.to(device)}, \
           label, labels.to(device), num_classes, \
           train_mask.to(device), val_mask.to(device), test_mask.to(device)


def load_imdb(remove_self_loop):
    from openhgnn import GTNDataset
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataset = GTNDataset(name='imdb4GTN')
    hg = dataset[0]
    features = hg.ndata['h']
    movie_features = features['movie']
    actor_features = features['actor']
    director_features = features['director']
    labels = hg.ndata['label']['movie']
    train_mask = hg.ndata['train_mask']['movie']
    val_mask = hg.ndata['val_mask']['movie']
    test_mask = hg.ndata['test_mask']['movie']
    num_classes = 3
    label = 'movie'
    etypes = ['actor-movie', 'movie-actor', 'movie-director', 'director-movie']
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    return hg.to(device), etypes, {'movie': movie_features.to(device), 'actor': actor_features.to(device),
                                   "director": director_features.to(device)}, \
           label, labels.to(device), num_classes, \
           train_mask.to(device), val_mask.to(device), test_mask.to(device)


# def load_ogb(remove_self_loop):
#     from ogb.nodeproppred import DglNodePropPredDataset
#
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#     dataset = DglNodePropPredDataset(name='ogbn-mag')
#     split_idx = dataset.get_idx_split()
#     train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
#     hg, labels_t = dataset[0]
#
#     labels = torch.squeeze(labels_t['paper'])
#     choice = [i for i, x in enumerate(labels) if x == 1 or x == 134]
#     choice = np.array(choice)
#     # num_catgory = hg.number_of_nodes('paper')
#     # catgory_numpy = np.arange(num_catgory)
#     # choice = np.random.choice(catgory_numpy, int(len(catgory_numpy) * 0.01))
#     # np.save('choice.txt', choice)
#     # choice = np.load('choice.txt')
#     hg = dgl.sampling.sample_neighbors(hg, {'paper': choice}, 1)
#     hg = dgl.node_type_subgraph(hg, ['author', 'paper'])
#
#     new_labels = [int(0) if i==134 else int(i) for i in labels]
#     labels = torch.tensor(new_labels)
#
#     train_mask = np.zeros(hg.number_of_nodes('paper'))
#     val_mask = np.zeros(hg.number_of_nodes('paper'))
#     test_mask = np.zeros(hg.number_of_nodes('paper'))
#     paper_features = hg.ndata['feat']['paper']
#     author_features = torch.rand(hg.number_of_nodes('author'), paper_features.shape[1])
#     # field_of_study_features = torch.rand(hg.number_of_nodes('field_of_study'), paper_features.shape[1])
#     # institution_features = torch.rand(hg.number_of_nodes('institution'), paper_features.shape[1])
#     # etypes = ['affiliated_with', 'writes', 'cites', 'has_topic']
#
#     etypes = ['writes', 'cites']
#     num_classes = 2
#     print(hg)
#     for train_id in train_idx['paper'].numpy():
#         if choice.__contains__(train_id):
#             train_mask[train_id] = 1
#     for val_id in valid_idx['paper'].numpy():
#         if choice.__contains__(val_id):
#             val_mask[val_id] = 1
#     for test_id in test_idx['paper'].numpy():
#         if choice.__contains__(test_id):
#             test_mask[test_id] = 1
#
#     # np.save('train_mask.txt', train_mask)
#     # np.save('val_mask.txt', val_mask)
#     # np.save('test_mask.txt', test_mask)
#
#     # train_mask = np.load('train_mask.txt')
#     # val_mask = np.load('val_mask.txt')
#     # test_mask = np.load('test_mask.txt')
#
#     if hasattr(torch, 'BoolTensor'):
#         train_mask = torch.tensor(train_mask).bool()
#         val_mask = torch.tensor(val_mask).bool()
#         test_mask = torch.tensor(test_mask).bool()
#
#     # from collections import Counter
#     # res = Counter(labels[train_mask].tolist())
#     # print(res)
#     print(hg)
#     print(author_features.shape)
#     return hg.to(device), etypes, {'author': author_features.to(device), 'paper': paper_features.to(device)}, \
#            'paper', labels.to(device), num_classes, \
#            train_mask.to(device), val_mask.to(device), test_mask.to(device)

def load_ogb(remove_self_loop):
    from ogb.nodeproppred import DglNodePropPredDataset

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dataset = DglNodePropPredDataset(name='ogbn-mag')
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"]['paper'], split_idx["valid"]['paper'], split_idx["test"]['paper']
    hg, labels = dataset[0]

    features = hg.nodes['paper'].data['feat']
    hg.nodes["paper"].data["feat"] = features


    labels = labels['paper'].to(device).squeeze()
    n_classes = int(labels.max() - labels.min()) + 1

    target_node_type = "paper"
    feature_node_types = [target_node_type]

    print(n_classes)
    print(feature_node_types)
    print(features.shape)
    # return hg.to(device), etypes, {'author': author_features.to(device), 'paper': paper_features.to(device)}, \
    #        'paper', labels.to(device), num_classes, \
    #        train_mask.to(device), val_mask.to(device), test_mask.to(device)

def load_amazon(remove_self_loop):
    from openhgnn import HGBDataset
    import scipy
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    emd_size = 128
    path = ['product-product-0', 'product-product-1']
    test_choice = 0.7
    val_choice = 0.1
    train_choice = 0.2
    dataset = HGBDataset(name='HGBl-amazon')
    hg = dataset[0]
    features = hg.ndata['h']

    test_pos_u_dict = {}
    test_pos_v_dcit = {}
    val_pos_u_dict = {}
    val_pos_v_dict = {}
    train_pos_u_dict = {}
    train_pos_v_dict = {}

    test_neg_u_dict = {}
    test_neg_v_dict = {}
    val_neg_u_dict = {}
    val_neg_v_dict = {}
    train_neg_u_dict = {}
    train_neg_v_dict = {}
    for p in path:
        u, v = hg.edges(etype=p)
        eids = np.arange(hg.number_of_edges(etype=p))

        eids = np.random.permutation(eids)
        test_size = int(len(eids) * test_choice)
        train_size = int(len(eids) * train_choice)
        val_size = int(len(eids) * val_choice)
        test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
        test_pos_u_dict[p] = test_pos_u
        test_pos_v_dcit[p] = test_pos_v

        val_pos_u, val_pos_v = u[eids[test_size:test_size + val_size]], v[eids[test_size:test_size + val_size]]
        val_pos_u_dict[p] = val_pos_u
        val_pos_v_dict[p] = val_pos_v

        train_pos_u, train_pos_v = u[eids[test_size + val_size:]], v[eids[test_size + val_size:]]
        train_pos_u_dict[p] = train_pos_u
        train_pos_v_dict[p] = train_pos_v

        adj = scipy.sparse.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
        try:
            adj_neg = 1 - adj.todense() - np.eye(hg.number_of_nodes('product'))
        except:
            adj_neg = 1 - adj.todense()
        neg_u, neg_v = np.where(adj_neg != 0)

        neg_eids = np.random.choice(len(neg_u), hg.number_of_edges(etype=p))
        test_neg_u, test_neg_v = (
            neg_u[neg_eids[:test_size]],
            neg_v[neg_eids[:test_size]],
        )
        test_neg_u_dict[p] = test_neg_u
        test_neg_v_dict[p] = test_neg_v
        val_neg_u, val_neg_v = (
            neg_u[neg_eids[test_size:test_size + val_size]],
            neg_v[neg_eids[test_size:test_size + val_size]],
        )
        val_neg_u_dict[p] = val_neg_u
        val_neg_v_dict[p] = val_neg_v

        train_neg_u, train_neg_v = (
            neg_u[neg_eids[test_size + val_size:]],
            neg_v[neg_eids[test_size + val_size:]],
        )
        train_neg_u_dict[p] = train_neg_u
        train_neg_v_dict[p] = train_neg_v
        hg = dgl.remove_edges(hg, eids[:test_size + val_size], p)
    train_pos_hg = dgl.heterograph({
        (p.split('-')[0], p, p.split('-')[1]): (train_pos_u_dict[p], train_pos_v_dict[p]) for p in
        train_pos_u_dict.keys()
    }, num_nodes_dict={'product': hg.number_of_nodes('product')})
    train_neg_hg = dgl.heterograph({
        (p.split('-')[0], p, p.split('-')[1]): (train_neg_u_dict[p], train_neg_v_dict[p]) for p in
        train_neg_u_dict.keys()
    }, num_nodes_dict={'product': hg.number_of_nodes('product')})

    val_pos_hg = dgl.heterograph({
        (p.split('-')[0], p, p.split('-')[1]): (val_pos_u_dict[p], val_pos_v_dict[p]) for p in
        val_pos_u_dict.keys()
    }, num_nodes_dict={'product': hg.number_of_nodes('product')})
    val_neg_hg = dgl.heterograph({
        (p.split('-')[0], p, p.split('-')[1]): (val_neg_u_dict[p], val_neg_v_dict[p]) for p in
        val_neg_u_dict.keys()
    }, num_nodes_dict={'product': hg.number_of_nodes('product')})

    test_pos_hg = dgl.heterograph({
        (p.split('-')[0], p, p.split('-')[1]): (test_pos_u_dict[p], test_pos_v_dcit[p]) for p in test_pos_u_dict.keys()
    }, num_nodes_dict={'product': hg.number_of_nodes('product')})
    test_neg_hg = dgl.heterograph({
        (p.split('-')[0], p, p.split('-')[1]): (test_neg_u_dict[p], test_neg_v_dict[p]) for p in test_neg_u_dict.keys()
    }, num_nodes_dict={'product': hg.number_of_nodes('product')})
    label = 'product'
    return path, label, True, hg.to(device), {'product': features.to(device)}, train_pos_hg.to(device), train_neg_hg.to(
        device), val_pos_hg.to(device), val_neg_hg.to(device), test_pos_hg.to(device), test_neg_hg.to(device)


def load_lastfm(remove_self_loop):
    from openhgnn import HGBDataset
    import scipy
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    emd_size = 128
    path = ['artist-tag', 'user-artist', 'user-user']
    test_choice = 0.7
    val_choice = 0.1
    train_choice = 0.2
    dataset = HGBDataset(name='HGBl-LastFM')
    hg = dataset[0]
    artist_features = torch.rand(hg.number_of_nodes('artist'), emd_size)
    user_features = torch.rand(hg.number_of_nodes('user'), emd_size)
    tag_features = torch.rand(hg.number_of_nodes('tag'), emd_size)

    test_pos_u_dict = {}
    test_pos_v_dcit = {}
    val_pos_u_dict = {}
    val_pos_v_dict = {}
    train_pos_u_dict = {}
    train_pos_v_dict = {}

    test_neg_u_dict = {}
    test_neg_v_dict = {}
    val_neg_u_dict = {}
    val_neg_v_dict = {}
    train_neg_u_dict = {}
    train_neg_v_dict = {}
    for p in path:
        u, v = hg.edges(etype=p)
        eids = np.arange(hg.number_of_edges(etype=p))
        eids = np.random.permutation(eids)
        test_size = int(len(eids) * test_choice)
        train_size = int(len(eids) * train_choice)
        val_size = int(len(eids) * val_choice)
        test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
        test_pos_u_dict[p] = test_pos_u
        test_pos_v_dcit[p] = test_pos_v

        val_pos_u, val_pos_v = u[eids[test_size:test_size + val_size]], v[eids[test_size:test_size + val_size]]
        val_pos_u_dict[p] = val_pos_u
        val_pos_v_dict[p] = val_pos_v

        train_pos_u, train_pos_v = u[eids[test_size + val_size:]], v[eids[test_size + val_size:]]
        train_pos_u_dict[p] = train_pos_u
        train_pos_v_dict[p] = train_pos_v

        adj = scipy.sparse.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
        if p.split('-')[0] == p.split('-')[1]:
            adj_neg = 1 - adj.todense() - np.eye(hg.number_of_nodes(p.split('-')[0]))
        else:
            adj_neg = 1 - adj.todense()
        neg_u, neg_v = np.where(adj_neg != 0)

        neg_eids = np.random.choice(len(neg_u), hg.number_of_edges(etype=p))
        test_neg_u, test_neg_v = (
            neg_u[neg_eids[:test_size]],
            neg_v[neg_eids[:test_size]],
        )
        test_neg_u_dict[p] = test_neg_u
        test_neg_v_dict[p] = test_neg_v
        val_neg_u, val_neg_v = (
            neg_u[neg_eids[test_size:test_size + val_size]],
            neg_v[neg_eids[test_size:test_size + val_size]],
        )
        val_neg_u_dict[p] = val_neg_u
        val_neg_v_dict[p] = val_neg_v

        train_neg_u, train_neg_v = (
            neg_u[neg_eids[test_size + val_size:]],
            neg_v[neg_eids[test_size + val_size:]],
        )
        train_neg_u_dict[p] = train_neg_u
        train_neg_v_dict[p] = train_neg_v
        hg = dgl.remove_edges(hg, eids[:test_size + val_size], p)
    train_pos_hg = dgl.heterograph({
        (p.split('-')[0], p, p.split('-')[1]): (train_pos_u_dict[p], train_pos_v_dict[p]) for p in
        train_pos_u_dict.keys()
    }, num_nodes_dict={'artist': hg.number_of_nodes('artist'), 'user': hg.number_of_nodes('user'),
                       'tag': hg.number_of_nodes('tag')})
    train_neg_hg = dgl.heterograph({
        (p.split('-')[0], p, p.split('-')[1]): (train_neg_u_dict[p], train_neg_v_dict[p]) for p in
        train_neg_u_dict.keys()
    }, num_nodes_dict={'artist': hg.number_of_nodes('artist'), 'user': hg.number_of_nodes('user'),
                       'tag': hg.number_of_nodes('tag')})

    val_pos_hg = dgl.heterograph({
        (p.split('-')[0], p, p.split('-')[1]): (val_pos_u_dict[p], val_pos_v_dict[p]) for p in
        val_pos_u_dict.keys()
    }, num_nodes_dict={'artist': hg.number_of_nodes('artist'), 'user': hg.number_of_nodes('user'),
                       'tag': hg.number_of_nodes('tag')})
    val_neg_hg = dgl.heterograph({
        (p.split('-')[0], p, p.split('-')[1]): (val_neg_u_dict[p], val_neg_v_dict[p]) for p in
        val_neg_u_dict.keys()
    }, num_nodes_dict={'artist': hg.number_of_nodes('artist'), 'user': hg.number_of_nodes('user'),
                       'tag': hg.number_of_nodes('tag')})

    test_pos_hg = dgl.heterograph({
        (p.split('-')[0], p, p.split('-')[1]): (test_pos_u_dict[p], test_pos_v_dcit[p]) for p in test_pos_u_dict.keys()
    }, num_nodes_dict={'artist': hg.number_of_nodes('artist'), 'user': hg.number_of_nodes('user'),
                       'tag': hg.number_of_nodes('tag')})
    test_neg_hg = dgl.heterograph({
        (p.split('-')[0], p, p.split('-')[1]): (test_neg_u_dict[p], test_neg_v_dict[p]) for p in test_neg_u_dict.keys()
    }, num_nodes_dict={'artist': hg.number_of_nodes('artist'), 'user': hg.number_of_nodes('user'),
                       'tag': hg.number_of_nodes('tag')})
    label = 'artist'
    return path, label, False, hg.to(device), {'artist': artist_features.to(device), 'user': user_features.to(device),
                                               'tag': tag_features.to(
                                                   device)}, train_pos_hg.to(device), train_neg_hg.to(
        device), val_pos_hg.to(device), val_neg_hg.to(device), test_pos_hg.to(device), test_neg_hg.to(device)
