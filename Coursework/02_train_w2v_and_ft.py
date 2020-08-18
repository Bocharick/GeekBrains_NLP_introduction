import time
import gensim
from gensim.models import Word2Vec, FastText
import pickle
import glob
import multiprocessing

start_time = time.time()

# pkl_files_list = [os.path.join(os.path.abspath("../data/Otvety.txt_processed"), pkl_file) for pkl_file in os.listdir("../data/Otvety.txt_processed")]
pkl_files_list = glob.glob("../data/Otvety.txt_processed/*.pkl")

sentences = []
# for pkl_filepath in pkl_files_list[:1]:
for pkl_filepath in pkl_files_list:
    with open(pkl_filepath, "rb") as fin:
        sentences += pickle.load(fin)

# sentences = sentences[:100000]
print("sentences length:", len(sentences))

gensim_version = gensim.__version__
modelW2V_filepath = "../data/word2vec_ru_gensim_%s.model" % gensim_version
modelFT_filepath = "../data/fasttext_ru_gensim_%s.model" % gensim_version

train_time = time.time()
modelW2V = Word2Vec(sentences=sentences, size=320, window=7, min_count=10, workers=multiprocessing.cpu_count())
# modelW2V = Word2Vec(sentences=sentences, size=10, window=7, min_count=10, workers=multiprocessing.cpu_count())
print("Word2Vec trained for %.2f seconds." % (time.time() - train_time))
modelW2V.save(modelW2V_filepath)

train_time = time.time()
modelFT = FastText(sentences=sentences, size=320, min_count=10, window=7, workers=multiprocessing.cpu_count())
# modelFT = FastText(sentences=sentences, size=10, min_count=10, window=7, workers=multiprocessing.cpu_count())
print("FastText trained for %.2f seconds." % (time.time() - train_time))
modelFT.save(modelFT_filepath)

# print(modelFT.wv[word])

print("\nTime: %.2f seconds" % (time.time() - start_time))
