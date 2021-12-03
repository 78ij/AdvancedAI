import sys
import os
import papermill as pm
import scrapbook as sb
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.get_logger().setLevel('ERROR') # only show error messages

from recommenders.utils.timer import Timer
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.models.deeprec.deeprec_utils import prepare_hparams

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Tensorflow version: {}".format(tf.__version__))

# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# Model parameters
EPOCHS = 50
BATCH_SIZE = 1024

SEED = DEFAULT_SEED  # Set None for non-deterministic results

yaml_file = "./lightgcn.yaml"
user_file = "../../tests/resources/deeprec/lightgcn/user_embeddings.csv"
item_file = "../../tests/resources/deeprec/lightgcn/item_embeddings.csv"


df = pd.read_csv('./datasets/train_dataset.csv')

df = df.rename(columns={'user_id':'userID','item_id':'itemID'})
df['rating'] = [1 for i in range(df.shape[0])]
print(df.head())

train, test = python_stratified_split(df, ratio=0.90)
data = ImplicitCF(train=train, test=test, seed=SEED)
hparams = prepare_hparams(yaml_file,
                          n_layers=3,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          learning_rate=0.005,
                          eval_epoch=5,
                          top_k=TOP_K,
                         )
model = LightGCN(hparams, data, seed=SEED)
model.load('E:\\AdvancedAI\\lightgcn_model\\epoch_13')

with Timer() as train_time:
    model.fit()