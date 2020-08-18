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
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

start_time = time.time()
gensim_version = gensim.__version__

# model_type = "W2V"
# OR
model_type = "FT"

if model_type == "W2V":
    modelW2V_filepath = "../data/word2vec_ru_gensim_%s.model" % gensim_version
    modelW2V = gensim.models.Word2Vec.load(modelW2V_filepath)
    modelW2V_vector_size = len(modelW2V.wv[random.choice(list(modelW2V.wv.vocab.keys()))])
    # print("modelW2V vector size:", modelW2V_vector_size)
    w2v_index = annoy.AnnoyIndex(modelW2V_vector_size, 'angular')
    w2v_index.load("../data/word2vec_annoy_index.ann")
    _index = w2v_index
    _model = modelW2V
    _veclen = modelW2V_vector_size
elif model_type == "FT":
    modelFT_filepath = "../data/fasttext_ru_gensim_%s.model" % gensim_version
    modelFT = gensim.models.FastText.load(modelFT_filepath)
    modelFT_vector_size = len(modelFT.wv[random.choice(list(modelFT.wv.vocab.keys()))])
    # print("modelFT vector size:", modelFT_vector_size)
    ft_index = annoy.AnnoyIndex(modelFT_vector_size, 'angular')
    ft_index.load("../data/fasttext_annoy_index.ann")
    _index = ft_index
    _model = modelFT
    _veclen = modelFT_vector_size
else:
    print("ERROR: Wrong model_type")
    exit()

morpher = MorphAnalyzer()
sw = set(get_stop_words("ru"))
exclude = set(string.punctuation)

with open("../data/index_map.pkl", "rb") as fin:
    index_map = pickle.load(fin)

print("Бот Анатолий загрузился за %.2f секунд, и готов к беседе.\n" % (time.time() - start_time))

# username = input("Представьтесь, пожалуйста: ")
# print()
# print("*" * 75, "Задавайте свои вопросы, или поддерживайте беседу. Наберите \'хватит\' для конца диалога:", "*" * 75,
#       sep="\n")
# while True:
#     print("\n%s: " % username, end="")
#     TEXT = input()
#     if TEXT.lower() == "хватит":
#         print("\nАнатолий: До свидания! Приятно было пообщаться")
#         break
#
#     _answer = get_response(TEXT, _index, _model, index_map, morpher, sw, exclude, _veclen)
#     print("\nАнатолий %s: " % model_type, random.choice(_answer).strip(), flush=True)


updater = Updater(token='1344465406:AAH1HA-yxdE2C_MYC_BOF0ZuydQXm2G8jbU')  # Токен API к Telegram
dispatcher = updater.dispatcher


def startCommand(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text='Добрый день')


def textMessage(bot, update):
    response = 'Ваше сообщение принял ' + update.message.text  # формируем текст ответа
    bot.send_message(chat_id=update.message.chat_id, text=response)


def startCommand(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text='Добрый день')


def textMessage(bot, update):
    TEXT = update.message.text
    if TEXT == "абырвалг":
        bot.send_message(chat_id=update.message.chat_id, text='откуда ты знаешь пароль, падла?')
    else:
        _answer = get_response(TEXT, _index, _model, index_map, morpher, sw, exclude, _veclen)

        if _answer:
            bot.send_message(chat_id=update.message.chat_id, text=random.choice(_answer).strip())
        else:
            bot.send_message(chat_id=update.message.chat_id, text='что?')


# Хендлеры
start_command_handler = CommandHandler('start', startCommand)
text_message_handler = MessageHandler(Filters.text, textMessage)
# Добавляем хендлеры в диспетчер
dispatcher.add_handler(start_command_handler)
dispatcher.add_handler(text_message_handler)
# Начинаем поиск обновлений
updater.start_polling(clean=True)
# Останавливаем бота, если были нажаты Ctrl + C
updater.idle()
