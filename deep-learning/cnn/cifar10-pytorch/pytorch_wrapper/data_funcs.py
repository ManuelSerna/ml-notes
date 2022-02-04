# Data-related helper functions
# (e.g., managing directories with data, .pth files, etc.)
# for Pytorch projects
# ALIAS: data_funcs

import csv
import json


def write_dict_data(data=None, filename=None):
    ''' Write data from dictionary generated from training to file

    :param data: (dictionary) data to write to file
    :param filename: name of file to write to,
        can be .csv or .json
    :return: NA
    '''
    if filename[-3:] == 'csv':
        raise ValueError('Writing to csvs not implemented yet...')
        print('[NOTICE] Writing to csv.')

        field_names = [key for key, value in data.items()]

        with open(filename, 'w') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=field_names)
            csvwriter.writeheader()
            csvwriter.writerows(data)
    elif filename[-4:] == 'json':
        print('[NOTICE] Writing to json.')
        with open(filename, 'w') as file:
            file.write(json.dumps(data, indent=4))
    else:
        raise ValueError('File extension not supported!')


def read_dict_json(filename=''):
    ''' Read in dictionary (JSON object) from JSON file

    :param filename: name of file to read
    :return: dictionary
    '''
    txt = open(filename, 'r')
    txt_str = txt.read()
    d = json.loads(txt_str)
    txt.close()
    return d
