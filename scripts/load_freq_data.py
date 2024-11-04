import csv
from scripts.config import Config
params = Config().params



class Data:
    def __init__(self, hpo_id, name, freq, propagated_freq):
        self.hpo_id = hpo_id
        self.name = name
        self.freq = freq
        self.propagated_freq = propagated_freq

def read_csv(file_path, has_header = True):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        if has_header:
            header = next(reader)
        for row in reader:
            data.append(Data(row[0], row[1], row[2], row[3]))
    return data

def load_freq_data(csv_file_path):
    frequecy_data = read_csv(csv_file_path)
    frequecy_dict = {}
    for item in frequecy_data:
        frequecy_dict[item.hpo_id]={ 'name': item.name, 'freq': item.freq, 'propagated_freq': item.propagated_freq}   
    return frequecy_dict
