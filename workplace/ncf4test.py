import sys
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel('ERROR')  # only show error messages

from reco_utils.common.timer import Timer
from reco_utils.recommender.ncf.ncf_singlenode import NCF
from reco_utils.recommender.ncf.dataset import Dataset as NCFDataset
from reco_utils.dataset import movielens
from reco_utils.common.notebook_utils import is_jupyter
from reco_utils.dataset.python_splitters import python_chrono_split, python_stratified_split, python_random_split
from reco_utils.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k,
                                                     recall_at_k, get_top_k_items)

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Tensorflow version: {}".format(tf.__version__))

# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# Model parameters
EPOCHS = 100
BATCH_SIZE = 256

SEED = 0

# pandas donwload
df = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=["userID", "itemID", "rating"]
)
df = df[:1000]
train, test = python_stratified_split(df, 0.75)

data = NCFDataset(train=train, test=test, seed=SEED)
# for row in data.test_loader():
gmf = NCF(
    n_users=data.n_users,
    n_items=data.n_items,
    model_type="GMF",
    n_factors=4,
    layer_sizes=[16, 8, 4],
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=10,
    seed=SEED
)
with Timer() as train_time:
    gmf.fit(data)

print("Took {} seconds for training.".format(train_time.interval))

gmf.save(dir_name=".pretrain/GMF")
mlp = NCF(
    n_users=data.n_users,
    n_items=data.n_items,
    model_type="MLP",
    n_factors=4,
    layer_sizes=[16, 8, 4],
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=10,
    seed=SEED
)
with Timer() as train_time:
    mlp.fit(data)

print("Took {} seconds for training.".format(train_time.interval))

mlp.save(dir_name=".pretrain/MLP")
model = NCF(
    n_users=data.n_users,
    n_items=data.n_items,
    model_type="NeuMF",
    n_factors=4,
    layer_sizes=[16, 8, 4],
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=10,
    seed=SEED
)
model.load(gmf_dir=".pretrain/GMF", mlp_dir=".pretrain/MLP", alpha=0.5)
with Timer() as test_time:

    users, items, preds = [], [], []
    item = list(train.itemID.unique())
    for user in train.userID.unique():
        user = [user] * len(item)
        users.extend(user)
        items.extend(item)
        preds.extend(list(model.predict(user, item, is_list=True)))

    all_predictions = pd.DataFrame(data={"userID": users, "itemID": items, "prediction": preds})

    merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
    all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

print("Took {} seconds for prediction.".format(test_time.interval))

# print("Took {} seconds for prediction.".format(test_time.interval))
# eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
#
# print("MAP:\t%f" % eval_map,
#       "NDCG:\t%f" % eval_ndcg,
#       "Precision@K:\t%f" % eval_precision,
#       "Recall@K:\t%f" % eval_recall, sep='\n')
#
# if is_jupyter():
#     # Record results with papermill for tests
#     import papermill as pm
#     import scrapbook as sb
#
#     sb.glue("map", eval_map)
#     sb.glue("ndcg", eval_ndcg)
#     sb.glue("precision", eval_precision)
#     sb.glue("recall", eval_recall)
#     sb.glue("train_time", train_time.interval)
#     sb.glue("test_time", test_time.interval)
# if __name__ == "__main__":
#     pass
