#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 12:08:25 2018

@author: ritesh
"""

#importing libraries
import os
import gensim
from nltk.corpus import movie_reviews 

#getting data
sentences = movie_reviews .sents()

#training word2vec model
model = gensim.models.Word2Vec(sentences, min_count=1)
#model.save('Mreview_model')

#load and test

#model = gensim.models.Word2Vec.load(‘Mreview_model’)
#words most similar to mother
print("Most Similar:",model.most_similar('mother'))

#find the odd one out
print(model.doesnt_match("breakfast cereal dinner lunch".split()))
print(model.doesnt_match("cat dog table".split()))

#vector representation of word human
print("Vecor representation of word human:",model['movie'])