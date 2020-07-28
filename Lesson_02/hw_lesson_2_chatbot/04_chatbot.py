import string
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
import pickle
import re
from collections import deque
import multiprocessing
import os
from botlib import preprocess_txt, count_textfile_lines_count
import gensim
import random
import annoy
from pymorphy2 import MorphAnalyzer
from stop_words import get_stop_words
import string
from botlib import preprocess_txt, count_textfile_lines_count, get_response
import numpy as np
import time
from gensim.models import Word2Vec, FastText
import glob

start_time = time.time()
gensim_version = gensim.__version__

modelW2V_filepath = "../data/word2vec_ru_gensim_%s.model" % gensim_version
modelFT_filepath = "../data/fasttext_ru_gensim_%s.model" % gensim_version

modelFT = gensim.models.FastText.load(modelFT_filepath)
modelW2V = gensim.models.Word2Vec.load(modelW2V_filepath)

modelW2V_vector_size = len(modelW2V.wv[random.choice(list(modelW2V.wv.vocab.keys()))])
# print("modelW2V vector size:", modelW2V_vector_size)

modelFT_vector_size = len(modelFT.wv[random.choice(list(modelFT.wv.vocab.keys()))])
# print("modelFT vector size:", modelFT_vector_size)


morpher = MorphAnalyzer()
sw = set(get_stop_words("ru"))
exclude = set(string.punctuation)

w2v_index = annoy.AnnoyIndex(modelW2V_vector_size, 'angular')
ft_index = annoy.AnnoyIndex(modelFT_vector_size, 'angular')

w2v_index.load("../data/word2vec_annoy_index.ann")
ft_index.load("../data/fasttext_annoy_index.ann")

with open("../data/index_map.pkl", "rb") as fin:
    index_map = pickle.load(fin)

print("Бот Анатолий загрузился за %.2f секунд, и готов к беседе.\n" % (time.time() - start_time))

username = input("Представьтесь, пожалуйста: ")
print()
print("*" * 75, "Задавайте свои вопросы, или поддерживайте беседу. Наберите \'хватит\' для конца диалога:", "*" * 75,
      sep="\n")
while True:
    TEXT = input("\n%s: " % username)
    if TEXT.lower() == "хватит":
        print("\nАнатолий: До свидания! Приятно было пообщаться")
        break

    answer_W2V = get_response(TEXT, w2v_index, modelW2V, index_map, morpher, sw, exclude, modelW2V_vector_size)
    print("\nАнатолий W2V: ", random.choice(answer_W2V).strip(), flush=True)

    answer_FT = get_response(TEXT, ft_index, modelFT, index_map, morpher, sw, exclude, modelFT_vector_size)
    print("\nАнатолий FT: ", random.choice(answer_FT).strip(), flush=True)
