import random
from torch import tensor
import pandas as pd

sentences = []

s1 = pd.read_csv('../../data/raw/medium_link_prediction_noClue_shuffled_train.csv', lineterminator="\n")
for i in range(len(s1)):
    item = s1.iloc[i]
    sentences.append((item['q1_Title'].strip() if type(item['q1_Title']) == str else '') + (item['q1_Body'].strip() if type(item['q1_Body']) == str else ''))
    sentences.append((item['q2_Title'].strip() if type(item['q2_Title']) == str else '') + (item['q2_Body'].strip() if type(item['q2_Body']) == str else ''))
# s2 = pd.read_csv('../../data/raw/medium_link_prediction_noClue_shuffled_test.csv', lineterminator="\n")
# for i in range(len(s2)):
#     item = s2.iloc[i]
#     sentences.append((item['q1_Title'].strip() if type(item['q1_Title']) == str else '') + (item['q1_Body'].strip() if type(item['q1_Body']) == str else ''))
#     sentences.append((item['q2_Title'].strip() if type(item['q2_Title']) == str else '') + (item['q2_Body'].strip() if type(item['q2_Body']) == str else ''))

l = len(sentences) * 8 // 10
random.shuffle(sentences)
with open('train.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(sentences[:l]))

with open('test.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(sentences[l:]))
