from __future__ import division
from __future__ import print_function

import time
import os
import sys

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import scipy.stats as stats

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

from sklearn import manifold
from scipy.special import expit

from optimizer import OptimizerVAE
from input_data import *
from model import *
from preprocessing import *

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seeded', 1, 'Set numpy random seed')
flags.DEFINE_integer('test_count', 10, 'Set num tests')
flags.DEFINE_integer('emb_size', 128, 'Number of eigenvectors for embedding')
flags.DEFINE_integer('connected_split', 1, 'use split with training set always connected')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if FLAGS.seeded:
    np.random.seed(1)

dataset_str = FLAGS.dataset

# Load data
adj, features = load_data(dataset_str)

adj_def = adj

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

rocs = np.zeros(FLAGS.test_count)
aps = np.zeros(FLAGS.test_count)

for k in range(FLAGS.test_count):

    func = get_test_edges
    if FLAGS.connected_split == 0:
        func = mask_test_edges

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = func(adj_def)
    val_edges = tuple(zip(*val_edges))
    val_edges_false = tuple(zip(*val_edges_false))
    test_edges = tuple(zip(*test_edges))
    test_edges_false = tuple(zip(*test_edges_false))
    adj = adj_train

    z = manifold.spectral_embedding(adj, n_components=FLAGS.emb_size, random_state=k)
    adj_rec = np.dot(z, z.T)

    preds = sigmoid(adj_rec[test_edges])
    preds_neg = sigmoid(adj_rec[test_edges_false])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    rocs[k] = roc_score
    aps[k] = ap_score

print((np.mean(rocs), stats.sem(rocs)))
print((np.mean(aps), stats.sem(aps)))  
