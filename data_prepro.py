import argparse
import numpy as np
import scipy.sparse as sp
from gen_ii_co_oc import load_sp_mat
from sklearn.preprocessing import normalize
import torch

def get_cmd():
    parser = argparse.ArgumentParser()
    #exp setting
    parser.add_argument("-d", "--dataset", default = "pog", type = str,help ="dataset to train" )
    args = parser.parse_args()
    return args

if __name__ =='__main__':
    paras = get_cmd().__dict__
    dataset_name = paras["dataset"]
    path_ibi = f"datasets/{dataset_name}/ibi_cooc.npz"
    path_cbc = f"datasets/{dataset_name}/cbc_cooc.npz"
    path_iui = f"datasets/{dataset_name}/iui_cooc.npz"
    path_bib = f"datasets/{dataset_name}/bib_cooc.npz"
    save_path_ibi = f"datasets/{dataset_name}/n_neigh_ibi"
    save_path_cbc = f"datasets/{dataset_name}/n_neigh_cbc"
    save_path_iui = f"datasets/{dataset_name}/n_neigh_iui"
    save_path_bib = f"datasets/{dataset_name}/n_neigh_bib"

    ibi = load_sp_mat(path_ibi)
    print("ibi edge:", ibi.getnnz())
    cbc = load_sp_mat(path_cbc)
    print("cbc edge:", cbc.getnnz())
    iui = load_sp_mat(path_iui)
    print("iui edge:", iui.getnnz())
    bib = load_sp_mat(path_bib)
    print("bib edge:", bib.getnnz())
    
    print("statistic")
    ii_b_max = int(ibi.max())
    print(f"max i-i interactions through b: {ii_b_max}")
    cc_b_max = int(cbc.max())
    print(f"max c-c interactions through u: {cc_b_max}")
    bb_i_max = int(bib.max())
    print(f"max b-b interactions through i: {bb_i_max}")
    ii_u_max = int(iui.max())
    print(f"max i-i interactions through u: {ii_u_max}")
    n_items = ibi.shape[0]
    n_bundles = bib.shape[0]
    n_cates = cbc.shape[0]
    # mask all diag weight
    diag_filter_i = sp.coo_matrix(
        (np.ones(n_items), ([i for i in range(0, n_items)], [i for i in range(0, n_items)])),
        shape=ibi.shape).tocsr()

    diag_filter_b = sp.coo_matrix(
        (np.ones(n_bundles), ([i for i in range(0, n_bundles)], [i for i in range(0, n_bundles)])),
        shape=bib.shape).tocsr()
    diag_filter_c = sp.coo_matrix(
        (np.ones(n_cates), ([i for i in range(0, n_cates)], [i for i in range(0, n_cates)])),
        shape=cbc.shape).tocsr()
    # mask all diag of filtered matrix
    diag_filter_iui = iui.multiply(diag_filter_i)
    diag_filter_ibi = ibi.multiply(diag_filter_i)
    diag_filter_bib = bib.multiply(diag_filter_b)
    diag_filter_cbc = cbc.multiply(diag_filter_c)

    neighbor_iui = iui - diag_filter_iui.tocsc()
    neighbor_ibi = ibi - diag_filter_ibi.tocsc()
    neighbor_bib = bib - diag_filter_bib.tocsc()
    neighbor_cbc = cbc - diag_filter_cbc.tocsc()

    n_iui = neighbor_iui.tocoo()
    n_ibi = neighbor_ibi.tocoo()
    n_bib = neighbor_bib.tocoo()
    n_cbc = neighbor_cbc.tocoo()
    
    ibi_edge_index = torch.tensor([list(n_ibi.row), list(n_ibi.col)], dtype=torch.int64)
    iui_edge_index = torch.tensor([list(n_iui.row), list(n_iui.col)], dtype=torch.int64)
    bib_edge_index = torch.tensor([list(n_bib.row), list(n_bib.col)], dtype=torch.int64)
    cbc_edge_index = torch.tensor([list(n_cbc.row), list(n_cbc.col)], dtype=torch.int64)

    # --------------------- saving --------------------------

    np.save(save_path_ibi, ibi_edge_index)
    np.save(save_path_iui, iui_edge_index)
    np.save(save_path_cbc, cbc_edge_index)
    np.save(save_path_bib, bib_edge_index)