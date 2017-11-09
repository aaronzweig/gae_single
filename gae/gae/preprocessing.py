import numpy as np
import scipy.sparse as sp
import networkx as nx


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph_coo(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

def preprocess_partials(adj):
    adj = sp.lil_matrix(adj)
    num_nodes = adj.shape[0]
    partials = []
    for i in range(num_nodes):
        # print(str(i) + '/' + str(num_nodes))
        partial = adj[:i, :i].tocoo()
        partial = sp.coo_matrix((partial.data, (partial.row, partial.col)), (num_nodes, num_nodes))
        partials.append(preprocess_graph_coo(partial))
    partials = sparse_to_tuple(sp.vstack(partials))
    return partials

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict

def pick_edges(graph, count):
    G = nx.Graph(graph)
    edges = []
    while len(edges) < count:
        G_edges = G.edges()
        i = np.random.randint(len(G_edges))
        u, v = G_edges[i]
        G.remove_edge(u, v)

        if True and nx.has_path(G, u, v):
            edges.append([u, v])
        else:
            G.add_edge(u, v)
    return edges

def pick_false_edges(graph, count):
    G = nx.Graph(graph)
    edges = []
    while len(edges) < count:
        G_nodes = G.nodes()
        i = np.random.randint(len(G_nodes))
        j = np.random.randint(len(G_nodes))
        u = G_nodes[i]
        v = G_nodes[j]

        if v not in G.neighbors(u) + [u]:
            edges.append([u, v])
            G.add_edge(u, v)
    return edges

def get_test_edges(adj):
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    edges_all = sparse_to_tuple(adj)[0]

    edge_count = adj.shape[0]
    num_test = int(np.floor(edge_count / 10.))
    num_val = int(np.floor(edge_count / 20.))

    G = nx.to_networkx_graph(adj)
    test_edges = pick_edges(G, num_test)
    test_edges_false = pick_false_edges(G, num_test)

    G.remove_edges_from(test_edges)
    val_edges = pick_edges(G, num_val)
    val_edges_false = pick_false_edges(G, num_val)

    G.remove_edges_from(val_edges)
    adj_train = nx.to_scipy_sparse_matrix(G)
    train_edges = sparse_to_tuple(adj_train)[0]

    def ismember(a, b):
        inter = [x for x in a if x in b]
        return len(inter) > 0

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)
    assert ismember(val_edges, val_edges)

    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false

def edge_dropout(adj, dropout):

    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    num_val = int(np.floor(edges.shape[0] * 1.0 * dropout))

    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    train_edges = np.delete(edges, val_edge_idx, axis=0)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    return adj_train

def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = range(edges.shape[0])
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return (np.all(np.any(rows_close, axis=-1), axis=-1) and
                np.all(np.any(rows_close, axis=0), axis=0))

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false