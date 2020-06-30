# -*- coding: utf-8 -*-
"""
문) 토큰 생성기를 이용해서 문장의 전체 단어 길이를 확인하고,
    희소행렬과 max length를 적용하여 문서단어 행렬을 만드시오. 
"""

from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    "Mr. Green killed Colonel Mustard in the study with the candlestick. Mr. Green is not a very nice fellow.",
    "Professor Plum has a green plant in his study.",
    "Miss Scarlett watered Professor Plum's green plant while he was away from his office last week."]

# 전체 문장 길이
print(len(sentences)) # 3


# 1. num_words 토큰 생성기 : 중요단어 50개 선정 
 


# 2. 전체 단어 개수 확인 



# 3. sparse matrix : count, freq, tfidf



# 4. max length 지정 문장 길이 맞춤 : 두번째 단어 길이 이용 


