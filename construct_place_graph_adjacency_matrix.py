from src.library.file_utils import *
from src.library.data_utils import *
from src.library.visual_utils import *
from src.library.geo_utils import *

import pandas as pd
import sys
import numpy
import time


#-----------
#  Data
#-----------

# data in linux
nyc_places = '~/PycharmProjects/adj_matrix_construction/data/place/nyc.csv'
chicago_places = '~/PycharmProjects/adj_matrix_construction/data/place/chicago.csv'
place_adj_matrix = '~/PycharmProjects/adj_matrix_construction/data/place/graph_adj_matrix.csv'
nyc_adj_matrix = '~/PycharmProjects/adj_matrix_construction/data/place/nyc_adj_matrix.csv'
chicago_adj_matrix = '~/PycharmProjects/adj_matrix_construction/data/place/chicago_adj_matrix.csv'
chicago_adj_matrix_with_distance = '~/PycharmProjects/adj_matrix_construction/data/place/chicago_adj_matrix_with_distance.csv'


# path in linux
merged_file = "~/PycharmProjects/Place_Category_Prediction/data/raw_data/merged_data_frame.csv"
nyc_places = "~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc.csv"
nyc_places_small = "~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc_small.csv"
nyc_places_with_id = "~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc_with_id.csv"
nyc_places_embedding_small = '~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc_embedding_small.csv'
nyc_small_adj_matrix = '~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc_small_adj_matrix.csv'
nyc_full_adj_matrix = '~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc_full_adj_matrix.csv'


#-------------------
# helper functions
#-------------------
def check_neighbor(origin_lat, origin_lon, target_lat, target_lon):
    dist = measure_distance(origin_lat, origin_lon, target_lat, target_lon)
    if dist < 0.01:
        return 1
    else:
        return 0


def save_adj_matrix(start_node_list, end_node_list, file_name):
    print('last index hit and save file')
    adj_matrix_df = pd.DataFrame(columns=['nyc_pid_int', 'neighbor_pid_int'])
    adj_matrix_df['nyc_pid_int'] = start_node_list
    adj_matrix_df['neighbor_pid_int'] = end_node_list
    adj_matrix_df.to_csv(file_name)


def get_NYC_adj_matrix():
    nyc_data = pd.read_csv(nyc_places)
    nyc_data['nyc_pid_int'] = pd.factorize(nyc_data.pid)[0]
    labels = ['nyc_pid_int', 'lat', 'lon', 'Level_1_Code']
    nyc_geo_data = nyc_data[labels]
    last_index = nyc_geo_data.shape[0] - 1

    # create graph dataframe
    graph_df = pd.DataFrame(columns=['nyc_pid_int', 'neighbor_pid_int'])
    start_node = []
    end_node = []
    start = time.time()
    for i in range(len(nyc_geo_data)):
        if i is last_index:
            save_adj_matrix(start_node, end_node, nyc_adj_matrix)
            break
        else:
            curr_lat = nyc_geo_data.loc[nyc_geo_data['nyc_pid_int'] == i]['lat']
            curr_lon = nyc_geo_data.loc[nyc_geo_data['nyc_pid_int'] == i]['lon']

            for j in range(i + 1, i + 20): # count 20 neighbor POIs
                if j > last_index:
                    save_adj_matrix(start_node, end_node, nyc_adj_matrix)
                    break
                else:
                    nei_lat = nyc_geo_data.loc[nyc_geo_data['nyc_pid_int'] == j]['lat']
                    nei_lon = nyc_geo_data.loc[nyc_geo_data['nyc_pid_int'] == j]['lon']
                    dist = measure_distance(curr_lat, curr_lon, nei_lat, nei_lon)
                    if dist < 0.07:
                        start_node.append(i)
                        end_node.append(j)

    graph_df['nyc_pid_int'] = start_node
    graph_df['neighbor_pid_int'] = end_node

    # save the connection in dataframe
    graph_df.to_csv(nyc_full_adj_matrix)
    end = time.time()


get_NYC_adj_matrix()



