import torch

from torch import nn
from util import load_mnist_traindataset, view_classification_mnist
from torch import optim
import torch.nn.functional as F
import numpy as np

# Making the results deterministic
torch.manual_seed(254)
np.random.seed(254)


db = torch.rand(5000) > 0.5
remove_index = 2

def get_parallel_db(db, remove_index):
    return torch.cat((db[0:remove_index], db[remove_index+1:]))

get_parallel_db(db, 1000)

def get_parallel_dbs(db):
    parallel_dbs = list()
    for i in range(len(db)):
        pdb = get_parallel_db(db, i)
        parallel_dbs.append(pdb)
    return parallel_dbs

pdbs = get_parallel_dbs(db)


def create_db_and_parallels(num_entries):
    # generate dbs and parallel dbs on the fly
    db = torch.rand(num_entries) > 0.5
    pdbs = get_parallel_dbs(db)

    return db, pdbs