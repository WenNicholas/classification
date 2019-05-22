# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
corpus = ['我 爱 中国 中国','爸爸 妈妈 爱 我','爸爸 妈妈 爱 中国']
# corpus = ['我爱中国','爸爸妈妈爱我','爸爸妈妈爱中国']
vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1)) ##创建词袋数据结构,里面相应参数设置
features = vectorizer.fit_transform(corpus)  #拟合模型，并返回文本矩阵

print("CountVectorizer:")
print(vectorizer.get_feature_names())   #显示所有文本的词汇，列表类型
print(vectorizer.vocabulary_)    #词汇表，字典类型
print(features)   #文本矩阵
print(features.toarray())   #.toarray() 是将结果转化为稀疏矩阵
print(features.toarray().sum(axis=0)) #统计每个词在所有文档中的词频

print()
print("TfidfVectorizer:")
vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=(1, 1))
features = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())   #显示所有文本的词汇，列表类型
print(vectorizer.vocabulary_)    #词汇表，字典类型
print(features)   #文本矩阵
print(features.toarray())   #.toarray() 是将结果转化为稀疏矩阵
print(features.toarray().sum(axis=0)) #统计每个词在所有文档中的词频