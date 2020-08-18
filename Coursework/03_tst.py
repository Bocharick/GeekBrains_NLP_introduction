import gensim
import random
import annoy
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
import string
from botlib import preprocess_txt, count_textfile_lines_count
import numpy as np
import pickle
import time

start_time = time.time()
gensim_version = gensim.__version__

modelW2V_filepath = "../data/word2vec_ru_gensim_%s.model" % gensim_version
modelFT_filepath = "../data/fasttext_ru_gensim_%s.model" % gensim_version

modelFT = gensim.models.FastText.load(modelFT_filepath)
modelW2V = gensim.models.Word2Vec.load(modelW2V_filepath)

modelW2V_vector_size = len(modelW2V.wv[random.choice(list(modelW2V.wv.vocab.keys()))])
print("modelW2V vector size:", modelW2V_vector_size)

modelFT_vector_size = len(modelFT.wv[random.choice(list(modelFT.wv.vocab.keys()))])
print("modelFT vector size:", modelFT_vector_size)

w2v_index = annoy.AnnoyIndex(modelW2V_vector_size, 'angular')
ft_index = annoy.AnnoyIndex(modelFT_vector_size, 'angular')

prepared_answers_filepath = "../data/prepared_answers.txt"
prepared_answers_lines_count = count_textfile_lines_count(prepared_answers_filepath)
print("File %s lines num:" % prepared_answers_filepath, prepared_answers_lines_count)

morpher = MorphAnalyzer()
sw = set(get_stop_words("ru"))
exclude = set(string.punctuation)

index_map = {}
counter = 0
with open(prepared_answers_filepath, "r") as f:
    timer1 = time.time()
    for line in f:
        n_w2v = 0
        n_ft = 0
        spls = line.split("\t")
        index_map[counter] = spls[1]
        question = preprocess_txt(spls[0], morpher, sw, exclude)

        vector_w2v = np.zeros(modelW2V_vector_size)
        vector_ft = np.zeros(modelFT_vector_size)
        for word in question:
            if word in modelW2V.wv:
                vector_w2v += modelW2V.wv[word]
                n_w2v += 1
            if word in modelFT.wv:
                vector_ft += modelFT.wv[word]
                n_ft += 1
        if n_w2v > 0:
            vector_w2v = vector_w2v / n_w2v
        if n_ft > 0:
            vector_ft = vector_ft / n_ft
        w2v_index.add_item(counter, vector_w2v)
        ft_index.add_item(counter, vector_ft)

        counter += 1
        if counter % 1764 == 0:
            speed = float(counter) / (time.time() - timer1)
            time_left = float(prepared_answers_lines_count - counter) / speed
            print("\rProgress: %d/%d lines complete; Speed: ~%.2f lines/second; Time left: ~%.2f seconds" % (counter, prepared_answers_lines_count, speed, time_left), end="")
        # if counter > 30000:
        #     break
    print("\rProgress: %d/%d lines complete; Speed: ~%.2f lines/second; Time left: ~%.2f seconds" % (counter, prepared_answers_lines_count, speed, time_left))
w2v_index.build(1000)
w2v_index.save("../data/word2vec_annoy_index.ann")

ft_index.build(1000)
ft_index.save("../data/fasttext_annoy_index.ann")

with open("../data/index_map.pkl", "wb") as fout:
    pickle.dump(index_map, fout)

print("\nTime: %.2f seconds" % (time.time() - start_time))


