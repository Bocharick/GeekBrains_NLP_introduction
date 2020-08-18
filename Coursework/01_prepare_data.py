import time
import string
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
import pickle
import re
from collections import deque
import multiprocessing
import os
from botlib import preprocess_txt, count_textfile_lines_count

start_time = time.time()

# otvety_filepath = "../data/Otvety55k.txt"
otvety_filepath = "../data/Otvety.txt"

# Small preprocess of the answers
question = None
written = False
with open("../data/prepared_answers.txt", "w") as fout:
    with open(otvety_filepath, "r") as fin:
        for line in fin:
            line = re.sub(r"<[^>]*>", " ", line)
            if line.startswith("---"):
                written = False
                continue
            if not written and question is not None:
                fout.write(question.replace("\t", " ").strip() + "\t" + line.replace("\t", " "))
                written = True
                question = None
                continue
            if not written:
                question = line.strip()
                continue


# Preprocess for models fitting

def process_file_part(args):
    st_time = time.time()
    filepath, n_lines, start_pos, is_last_part, output_dir = args
    if is_last_part == 1:
        n_lines += multiprocessing.cpu_count()
    file = open(filepath, "rt")
    file.seek(start_pos)

    morpher = MorphAnalyzer()
    sw = set(get_stop_words("ru"))
    exclude = set(string.punctuation)

    sentences = deque()

    for i in range(n_lines):
        line = file.readline()
        spls = preprocess_txt(line, morpher, sw, exclude)
        sentences.append(spls)

        if i % 764 == 0:
            speed = float(i) / (time.time() - start_time)
            print("Start pos: %d; Progress: %d/%d lines complete; Speed: ~%.2f lines/second" % (
                start_pos, i + 1, n_lines, speed))

    speed = float(i) / (time.time() - start_time)
    print(
        "Start pos: %d; Progress: %d/%d lines complete; Speed: ~%.2f lines/second" % (start_pos, i + 1, n_lines, speed))
    file.close()

    sentences = list(sentences)
    with open(os.path.join(output_dir, "preprocessed_sentences_list_%d.pkl" % start_pos), "wb") as sentences_list_file:
        pickle.dump(sentences, sentences_list_file)

cntr = count_textfile_lines_count(otvety_filepath)
print("File %s lines num:" % otvety_filepath, cntr)

num_cores = multiprocessing.cpu_count()
print("Num cores:", num_cores)

lines_in_one_part = cntr // num_cores
print("Lines in one part:", lines_in_one_part)
print()

print("Processing:")
start_poses = [(0, 0)]  # (start_pos, is_last_part)
fin = open(otvety_filepath, "rt")
for i in range(num_cores - 1):
    for j in range(lines_in_one_part):
        fin.readline()
    start_poses.append((fin.tell(), 0 if i < num_cores - 2 else 1))
fin.close()

output_dir = os.path.join(os.path.dirname(otvety_filepath), os.path.basename(otvety_filepath) + "_processed")
os.makedirs(output_dir, exist_ok=True)

generator = ((otvety_filepath, lines_in_one_part, start_pos, is_last_part, output_dir) for start_pos, is_last_part in
             start_poses)

pool = multiprocessing.Pool()
pool.map(process_file_part, generator)

print("\nTime: %.2f seconds" % (time.time() - start_time))
