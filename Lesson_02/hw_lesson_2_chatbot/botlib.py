import re
import numpy as np


def preprocess_txt(line, morpher, sw, exclude):
    line = re.sub(r"<[^>]*>", " ", line)
    spls = "".join(i for i in line.strip() if i not in exclude).split()
    spls = [morpher.parse(i.lower())[0].normal_form for i in spls]
    spls = [i for i in spls if i not in sw and i != ""]
    return spls


def count_textfile_lines_count(filepath):
    cntr = 0
    with open(filepath, "rt") as fin:
        for line in fin:
            cntr += 1
    # print("File %s lines num:" % filepath, cntr)
    return cntr


def get_response(question, index, model, index_map, morpher, sw, exclude, vec_len):
    question = preprocess_txt(question, morpher, sw, exclude)
    vector = np.zeros(vec_len)
    norm = 0
    for word in question:
        if word in model.wv:
            vector += model.wv[word]
            norm += 1
    if norm > 0:
        vector = vector / norm
    answers = index.get_nns_by_vector(vector, 3)
    return [index_map[i] for i in answers]

