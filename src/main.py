import json
from common.member import Member
from common.author import Author
from common.team import Team
from common.document import Document
from common.data_utils import *

counter = 0
docs = []
all_authors = {}  

training_input = []
training_output = []

data_path = "/home/ava1anche/Projects/TeamFormation/data/raw/dblp.v12.json"

all_authors, docs, training_input, training_output = read_data(data_path)

print(training_input[:3])
print(training_output[:3])