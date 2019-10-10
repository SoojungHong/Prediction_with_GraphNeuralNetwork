import pandas as pd
import sys
import numpy
import time


#----------------------------
#   data files in Linux
#----------------------------
merged_file = "~/PycharmProjects/Place_Category_Prediction/data/raw_data/merged_data_frame.csv"
nyc_places = "~/PycharmProjects/Place_Category_Prediction/data/raw_data/nyc.csv"
chicago_places = "~/PycharmProjects/Place_Category_Prediction/data/raw_data/chicago.csv"


#----------------------------
#  bounding box per city
#----------------------------
bbox_nyc = {'left_lon':-74.2591, 'bottom_lat':40.4774, 'right_lon':-73.7002, 'top_lat':40.9162}
bbox_chicago = {'left_lon':-88.133, 'bottom_lat':41.6062, 'right_lon':-87.4656, 'top_lat':42.1603}


#------------------------------
#  separate data per cities
#------------------------------
chicago_nyc_category_merged_data = pd.read_csv(merged_file)
nyc_dataframe = pd.DataFrame()
chicago_dataframe = pd.DataFrame()


# sort dataframe using 'lon' and 'lat'
sorted_chicago_nyc_category_merged_data = chicago_nyc_category_merged_data.sort_values(by=['lon', 'lat'])
nyc_data = sorted_chicago_nyc_category_merged_data[ sorted_chicago_nyc_category_merged_data['lon'] > bbox_nyc['left_lon']]


# set header
print(list(nyc_data.columns.values))
header = ['pid', 'place_name', 'category_id', 'lat', 'lon', 'category_name', 'Level_1_Code', 'Level_1_Name']
nyc_places_df = nyc_data[header]
nyc_places_df.to_csv(nyc_places)


