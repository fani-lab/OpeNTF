import sys
sys.path.extend(['../cmn'])

import json
from dal.data_utils import *

counter = 0
docs = []
all_authors = {}  

training_input = []
training_output = []

data_path = "../data/raw/dblp.v12.json"

all_authors, docs, training_input, training_output = read_data(data_path)

print(training_input[:3])
print(training_output[:3])