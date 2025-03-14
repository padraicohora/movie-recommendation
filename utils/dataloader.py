import os
from surprise import Reader, Dataset

file_path = os.path.expanduser('ml-100k/u.data')
print(file_path)

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format="user item rating timestamp", sep="\t")

_100k_data = Dataset.load_from_file(file_path, reader=reader)