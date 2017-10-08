"""
    Домашнее задание #2

    Требуется исправить опечатки в файле broken.csv; оценка качества --
    точное совпадение истинных и исправленных предложений.

    https://www.kaggle.com/c/csc-iinlp-2017-please-feax-me/

    Данный код использовать необязательно.

    Специализированные средства, разработанные специально для проверки правописания,
    использовать нельзя; в противном случае за работу будет выставлено 0 баллов.

    При этом использование библиотек с открытым исходным кодом, в которых реализованы
    edit distances (или даже language modeling), только приветствуется (и подразумевается).

    Если вы сомневаетесь, что можно использовать, а что нельзя -- спросите преподавателей.

    После того, как побиты бейзлайны, или если подходит к концу срок сдачи,
    нужно привести код в порядок и отправить его преподавателям через сайт,
    убедившись в приблизительной воспроизводимости результата на лидерборде.

    Побили ли вы бейзлайны "на привате", если уже побили их "на паблике", можно
    также уточнить у преподавателей.

    CSC, IINLP-2017

    P.S.: In real life, лучше писать в логи, а не выводить промежуточные результаты print-ами
    P.P.S.: Не откладывайте д/з в долгий ящик! Эта штука не так быстро считается (как могла бы).

"""
import codecs
import csv
import time
import nltk
from collections import Counter
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class Speller(object):
    """
        Поиск слов, наиболее близких по числу общих n-грамм и
        последующее ранжирование по эвристике-близости
    """

    def __init__(self, n_candidates_search=20):
        """
        :param n_candidates_search: число кандидатов-строк при поиске
        """
        # todo: может, это важный параметр?
        self.n_candidates = n_candidates_search

    def fit(self, words_list):
        """
            Подгонка спеллера
        """

        checkpoint = time.time()
        self.words_list = words_list

        # todo: может, что-то зависит от размера нграмм?
        # todo: может, надо работать не с бинарными значениями?
        self.vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(2, 2), binary=True)
        encoded_words = self.vectorizer.fit_transform(words_list).tocoo()

        self.index = defaultdict(set)

        # строим словарь, отображающий идентификатор нграммы в множество термов
        for i in zip(encoded_words.row, encoded_words.col):
            self.index[i[1]].add(i[0])

        print("Speller fitted in", time.time() - checkpoint)

        return self

    @lru_cache(maxsize=1000000)
    def rectify(self, word):
        """
            Предсказания спеллера
        """

        # запрос, преобразованный в нграммы
        char_ngrams_list = self.vectorizer.transform([word]).tocoo().col

        # для каждого терма считаем совпадение по нграммам
        counter = Counter()

        for token_id in char_ngrams_list:
            for word_id in self.index[token_id]:
                counter[word_id] += 1

        # ищем терм, ближайший по хитрому расстоянию из числа выбранных
        closest_word = word
        minimal_distance = 1000

        # среди топа по совпадениям по нграммам ищем "хорошее" исправление
        for suggest in counter.most_common(n=self.n_candidates):

            suggest_word = self.words_list[suggest[0]]

            # your code here
            s1 = set(suggest_word)
            s2 = set(word)
            distance = nltk.jaccard_distance(s1, s2)
            #distance = 0
            # можно использовать любые библиотеки и любые грязные хакерские трюки,
            # кроме использования источника текстов

            if distance < minimal_distance:
                minimal_distance = distance
                closest_word = suggest_word

        return closest_word


if __name__ == "__main__":

    np.random.seed(0)

    # зачитываем словарь "правильных слов"
    words_set = set(line.strip() for line in codecs.open("words2.txt", "r", encoding="utf-8"))
    words_list = sorted(list(words_set))

    # создаём спеллер
    speller = Speller()
    speller.fit(words_list)

    # читаем выборку
    df = pd.read_csv("broken.csv").head(500)
    # todo: замените строку выше на это, чтобы запустить генерацию решения на всём датасете
    # df = pd.read_csv("broken.csv")


    checkpoint1 = time.time()
    total_rectification_time = 0.0
    total_sentences_rectifications = 0.0

    y_submission = []
    counts = 0

    # исправляем, попутно собирая счётчики и засекая время
    for i in range(df.shape[0]):

        counts += 1

        if counts % 100 == 0:
            print("Rows processed", counts)

        start = time.time()
        mispelled_text = df["text"][i]
        mispelled_tokens = mispelled_text.split()

        was_rectified = False

        for j in range(len(mispelled_tokens)):
            if mispelled_tokens[j] not in words_set:
                rectified_token = speller.rectify(mispelled_tokens[j])
                mispelled_tokens[j] = rectified_token
                was_rectified = True

        if was_rectified:
            mispelled_text = " ".join(mispelled_tokens)
            total_rectification_time += time.time() - start
            total_sentences_rectifications += 1.0

        y_submission.append(mispelled_text)

    checkpoint2 = time.time()

    print("elapsed", checkpoint2 - checkpoint1)
    print("average speller time", total_rectification_time / float(total_sentences_rectifications))

    submission = pd.DataFrame({"id": df["id"], "text": y_submission}, columns=["id", "text"])
    submission.to_csv("baseline_submission.csv", index=None, encoding="utf-8", quotechar='"',
                      quoting=csv.QUOTE_NONNUMERIC)