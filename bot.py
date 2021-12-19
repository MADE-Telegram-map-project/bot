from collections import defaultdict
from typing import List
from time import sleep
import logging

import telebot
from telebot import types

from core.entities.data import MainConfig
from core.config import load_config
from core.ranking_model import Ranker

LOGGER = logging.getLogger()


path_to_config = "config.yaml"
config: MainConfig = load_config(path_to_config)
print(config)
bot = telebot.TeleBot(config.bot.token)
ranker = Ranker(config)

CHAN_SEARCH = "Поиск по каналу"
DESC_SEARCH = "Поиск по описанию"
RAND_SEARCH = "Случайный канал"
HELP = "Помощь"
MORE = "Еще каналы"

itembtn1 = types.KeyboardButton(CHAN_SEARCH)
itembtn2 = types.KeyboardButton(DESC_SEARCH)
itembtn3 = types.KeyboardButton(RAND_SEARCH)
itembtn4 = types.KeyboardButton(HELP)
itembtn5 = types.KeyboardButton(MORE)

markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
markup.add(itembtn1, itembtn2, itembtn3, itembtn4)

S0, S_CHAN, S_DESC = range(3)
user2state = defaultdict(int)  # dict of states of the conversations


def array2prety(array):
    return "\n\n".join(["@{} - {}".format(*row) for row in array])


@bot.message_handler(commands=['start'])
def send_welcome(message: types.Message):
    bot.send_message(
        message.chat.id, "Пришли мне ссылку на канал и я найду похожие на него")
    send_help(message)


@bot.message_handler(commands=['help'])
def send_help(message: types.Message):
    bot.send_message(
        message.chat.id,
        "Ссылка на канал может быть в формате:\n\t@channelname\n\tt.me/channelname\n\thttps://t.me/channelname\n\tchannelname",
        reply_markup=markup
    )


@bot.message_handler(regexp=CHAN_SEARCH)
def channel_search_choose(message: types.Message):
    bot.send_message(message.chat.id, "Укажи ссылку на канал")
    user2state[message.chat.id] = S_CHAN


@bot.message_handler(regexp=DESC_SEARCH)
def descr_search(message: types.Message):
    bot.send_message(message.chat.id, "Укажи описание, по которому я буду искать канал. Чем более подробным обо будет, тем лучше будет результат")
    user2state[message.chat.id] = S_DESC
    # chat_id = message.chat.id
    # bot.send_message(
    #     chat_id,
    #     "Извините, данная функциональность находится в разработке",
    #     reply_markup=markup
    # )
    # user2state[message.chat.id] = S0


@bot.message_handler(regexp=RAND_SEARCH)
def random_search(message: types.Message):
    chat_id = message.chat.id
    rand_channels = ranker.get_random_channels()
    bot.send_message(chat_id, "Вот 5 случайных каналов:")
    bot.send_message(
        chat_id,
        array2prety(rand_channels),
        reply_markup=markup,
    )
    user2state[message.chat.id] = S0


@bot.message_handler(regexp=HELP)
def help_button_press(message: types.Message):
    send_help(message)
    user2state[message.chat.id] = S0


@bot.message_handler(func=lambda message: user2state[message.chat.id] == S_CHAN)
def similar_channel_sending_chan(message: types.Message):
    sleep(5)
    chat_id = message.chat.id
    text = message.text
    top = ranker.get_channels_by_username(text)
    if top is None:
        bot.send_message(
            chat_id, "Пока что я не могу найти каналы, похожие на этот",
            reply_markup=markup
        )
    else:
        bot.send_message(chat_id, "Вот 5 каналов, похожих на данный:")
        bot.send_message(chat_id, array2prety(top))
        bot.send_message(
            chat_id,
            "Может быть еще что-нибудь найдем?",
            reply_markup=markup,
        )


@bot.message_handler(func=lambda message: user2state[message.chat.id] == S_DESC)
def similar_channel_sending_desc(message: types.Message):
    sleep(5)
    chat_id = message.chat.id
    text = message.text
    top = ranker.get_channels_by_description(text)
    if top is None:
        bot.send_message(
            chat_id, "По данному описанию я не могу найти ни одного канала, сорян",
            reply_markup=markup
        )
    else:
        bot.send_message(chat_id, "Вот каналы по описанию:")
        bot.send_message(chat_id, array2prety(top))
        bot.send_message(
            chat_id,
            "Может быть еще что-нибудь найдем?",
            reply_markup=markup,
        )


@bot.message_handler(func=lambda message: True)
def catch_all(message: telebot.types.Message):
    bot.send_message(
        message.chat.id,
        'Ничего не понял. Может лучше найдем интересный канал?',
        reply_markup=markup
    )
    user2state[message.chat.id] = S0


if __name__ == '__main__':
    LOGGER.info("Bot started...")
    bot.polling()
