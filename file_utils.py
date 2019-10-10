# libraries
import pandas as pd
import csv
import numpy as np
import time


# functions
def read_file_without_header(file):
    data = pd.read_csv(file, delimiter='\t')
    return data


def read_file(file, header):
    data = pd.read_csv(file, sep='\t', names=header, error_bad_lines=False)
    return data


def read_data_add_factorized_id(file, header):
    data = read_file(file, header)
    data['user_int'] = pd.factorize(data.cookie)[0]
    data['place_int'] = pd.factorize(data.result_name)[0]
    return data


def read_file_with_small_chunk(filename, chunk_size):
    import pandas as pd
    chunksize = chunk_size  # 10 ** 8
    file_index = 0;
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        save_file_name = filename + '_' + file_index + '.txt'
        print('save file name : ', save_file_name)
        chunk.to_csv(save_file_name, header=None, index=None, sep=' ', mode='a')
        file_index = file_index + 1


def get_header_names(file_name):
    df_chunk = pd.read_csv(file_name, chunksize=10)  # number of rows per chunk
    header = np.ndarray([])
    num_row = 1
    for chunk in df_chunk:
        if num_row == 1:
            header = chunk.columns.values
            return header
        else:
            num_row = num_row + 1


def read_small_chunk_from_file(file_name, chunk_size):
    df_chunk = pd.read_csv(file_name, chunksize=chunk_size)  # number of rows per chunk
    num_row = 1
    for chunk in df_chunk:
        if num_row == 1:
            return chunk
        else:
            num_row = num_row + 1


def read_big_csv_file(file_name):
    start = time.time()
    df = pd.read_csv(file_name)
    end = time.time()
    execution_time = end - start
    print("reading time in seconds : ", execution_time)
    return df


def divide_file_with_small_chunk(filename, write_filename, chunk_size):
    import pandas as pd
    chunksize = chunk_size
    file_index = 0;
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        save_file_name = write_filename + '_' + str(file_index) + '.csv'
        print('save file name : ', save_file_name)
        chunk.to_csv(save_file_name, header=True, index=None, sep=',', mode='a')
        file_index = file_index + 1
