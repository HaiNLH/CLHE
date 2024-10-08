import argparse
import numpy as np
import pandas as pd
import torch
import os
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from tqdm import tqdm
import json



def pairs2csr(pairs, shape):
    indice = np.array(pairs, dtype=np.int32)
    values = np.ones(len(pairs), dtype=np.float32)
    return sp.csr_matrix(
        (values, (indice[:, 0], indice[:, 1])), shape=shape)

def list2pairs(file):
    pairs = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            l = [int(i) for i in line.split(", ")]
            b_id = l[0]
            for i_id in l[1:]:
                pairs.append([b_id, i_id])
    return np.array(pairs)
def get_graph(path, x, y):
    pairs = list2pairs(path)

    indice = np.array(pairs, dtype=np.int32)
    values = np.ones(len(pairs), dtype=np.float32)
    graph = sp.csr_matrix(
        (values, (indice[:, 0], indice[:, 1])), shape=(x, y))
    print(pairs)
    return graph
def convert_sparse( sparse):
    dense_mat = sparse.toarray()
    dense_tensor= torch.tensor(dense_mat)
    return dense_tensor


def save_sp_mat(csr_mat, name):
    sp.save_npz(name, csr_mat)


def load_sp_mat(name):
    return sp.load_npz(name)


def filter(threshold, mat):
    mask = mat >= threshold
    mat = mat * mask
    return mat


def gen_ii_asym(ix_mat, threshold=0):
    '''
    mat: ui or bi
    '''
    ii_co = ix_mat @ ix_mat.T
    # i_count = ix_mat.sum(axis=1)
    # i_count += (i_count == 0) # mask all zero with 1
    # norm_ii = normalize(ii_asym, norm='l1', axis=1)
    # return norm_ii
    # return ii_asym
    mask = ii_co > threshold
    ii_co = ii_co.multiply(mask)
    # ii_asym = ii_co / i_count
    # normalize by row -> asym matrix
    return ii_co


def get_cmd():
    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("-d", "--dataset", default="pog", type=str, help="dataset to train")
    args = parser.parse_args()
    return args


def get_stat(path):
    with open(path, 'r') as f:
        stat = json.loads(f.read())
        
        print("done load json")
    print(stat)
    print(stat["#U"], stat["#B"], stat["#I"], stat["#C"])
    return stat["#U"], stat["#B"], stat["#I"], stat["#C"]



if __name__ == '__main__':

    dataset_name = "pog"
    data_path ='/content/drive/MyDrive/datasets'
    sep = ','
    print(f'{data_path}/{dataset_name}')
    users, bundles, items, cates = get_stat(f'{data_path}/{dataset_name}/count.json')
    dir = f'{data_path}/{dataset_name}'
    path = [dir + '/bi_train.txt',
            dir + '/item_cate.txt',
            dir + '/ui_full.txt']

    raw_graph = [get_graph(path[0], bundles, items),
                 get_graph(path[1], items, cates),
                 get_graph(path[2], users, items)]

    bi, ic, ui = raw_graph
    bc = bi @ ic
    pbar = tqdm(enumerate([bi.T, bc.T, ui.T, bi]), total=4, desc="gene", ncols=100)
    asym_mat = []
    for i, mat in pbar:
        asym_mat.append(gen_ii_asym(mat))

    pbar = tqdm(enumerate(["/ibi_cooc.npz", "/cbc_cooc.npz", "/iui_cooc.npz", "/bib_cooc.npz"]), total=4, desc="save",
                ncols=100)
    for i, data in pbar:
        save_sp_mat(asym_mat[i], dir + data)
