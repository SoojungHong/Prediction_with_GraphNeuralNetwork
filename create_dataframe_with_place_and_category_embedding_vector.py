import pandas as pd
import sys
import numpy as np
import time
import torch

from pytorch_transformers import *
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from math import *
from bert_serving.client import BertClient
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing


#-----------
#  Data
#-----------

# path in linux
merged_file = "~/PycharmProjects/Place_Category_Prediction/data/raw_data/merged_data_frame.csv"
nyc_places = "~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc.csv"
nyc_places_small = "~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc_small.csv"
nyc_places_with_id = "~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc_with_id.csv"
nyc_places_embedding_small = '~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc_embedding_small.csv'
nyc_places_embedding_full = '~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc_embedding_full.csv'


#----------------
#  Function
#----------------
def words_embedding(words):
    # start first the BERT service
    # linux > bert-serving-start -model_dir ~/Downloads/multi_cased_L-12_H-768_A-12/ -num_worker=4
    bc = BertClient()
    words_encoding = bc.encode([words])
    return words_encoding[0]


def _get_category_encoding(category_code, label_dictionary):
    c = label_dictionary[category_code]
    print('category encoding : ', c.tolist())  # [1 0 0] --> [1, 0, 0]
    return c


# one hot vector encoding for category
def get_category_encoding(category_code):
    if category_code == 100:
        ret = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    elif category_code == 200:
        ret = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])

    elif category_code == 300:
        ret = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])

    elif category_code == 350:
        ret = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])

    elif category_code == 400:
        ret = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

    elif category_code == 500:
        ret = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])

    elif category_code == 550:
        ret = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    elif category_code == 600:
        ret = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    elif category_code == 700:
        ret = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    elif category_code == 800:
        ret = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    else: #900
        ret = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    return ret


def create_nyc_dataset_prep():
    nyc_df = pd.read_csv(nyc_places)
    nyc_df = nyc_df.sort_values(by=['lat', 'lon'], ascending=True)

    # assign NYC specific pid_int
    nyc_df['nyc_pid_int'] = pd.factorize(nyc_df.pid)[0]
    nyc_df.columns = ['org_id', 'pid', 'place_name', 'category_id', 'lat', 'lon', 'category_name', 'Level_1_Code', 'Level_1_Name', 'nyc_pid_int']
    nyc_df.to_csv(nyc_places_with_id)

    # get embedding of place name
    embedding_start_time = time.time()

    nyc_df['place_embedding'] = nyc_df.apply(lambda row: words_embedding(row.place_name), axis=1)
    embedding_end_time = time.time()

    # Level 1 category to binary vector
    label_start_time = time.time()
    encoded_labels = LabelBinarizer().fit_transform(nyc_df['Level_1_Code'])  # make one hot vector of level 1
    label_dictionary = dict(zip(np.unique(nyc_df['Level_1_Code']).tolist(), np.unique(encoded_labels, axis=0)))  # dict(zip(['a', 'b', 'c'],[1, 2, 3]))

    nyc_df['label_encoding'] = nyc_df.apply(lambda row: get_category_encoding(row.Level_1_Code), axis=1)
    label_end_time = time.time()

    print('embedding elapsed time : ', embedding_end_time - embedding_start_time)
    print('labeling elapsed time : ', label_end_time - label_start_time)

    # save file
    nyc_df.to_csv(nyc_places_embedding_full)


create_nyc_dataset_prep()
