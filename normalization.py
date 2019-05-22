"""

@author: liushuchun
"""
import re
import string
import jieba

# 加载停用词
with open("dict/stop_words.utf8", encoding="utf8") as f:
    stopword_list = f.readlines()
# stopword_list = [line.rstrip() for line in open("dict/stop_words.utf8", 'r', encoding='utf-8')]

def tokenize_text(text):
    tokens = jieba.cut(text)
    tokens = [token.strip() for token in tokens]
    return tokens

#去除特殊符号
def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

#去停用词
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list] #这地方根本没有去停用词，如果去除了停用词，最后的评估模型的得分居然变低了。。。
    filtered_text = ''.join(filtered_tokens) #去掉空格
    return filtered_text


def normalize_corpus(corpus, tokenize=False):
    normalized_corpus = []
    # print(corpus)
    for text in corpus:

        text = remove_special_characters(text)
        text = remove_stopwords(text)
        normalized_corpus.append(text)
        # if tokenize:    #tokenize=True时执行
        #     text = tokenize_text(text)
        #     normalized_corpus.append(text)
    #仅仅返回的是去掉特殊符号的文本
    return normalized_corpus

