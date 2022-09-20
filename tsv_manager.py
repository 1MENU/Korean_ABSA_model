import pandas as pd
import csv
import os

df = pd.read_csv("dataset/task3_COPA/SKT_COPA_Train.tsv", sep="\t")

df = df[['ID', 'sentence', 'question', '1', '2', 'Answer']]

# df.rename(columns = {'SENTENCE1':'SENTENCE2', 'SENTENCE2':'SENTENCE1', 'start_s1':'start_s2', 'start_s2':'start_s1', 'end_s1':'end_s2', 'end_s2':'end_s1'}, inplace=True)

df.loc[df['question'] == "결과", 'question'] = "임시"
df.loc[df['question'] == "원인", 'question'] = "결과"
df.loc[df['question'] == "임시", 'question'] = "원인"

ans1 = df.loc[df['Answer'] == 1, '1']
ans2 = df.loc[df['Answer'] == 2, '2']

sent1 = df.loc[df['Answer'] == 1, 'sentence']
sent2 = df.loc[df['Answer'] == 2, 'sentence']

df.loc[df['Answer'] == 1, 'sentence'] = ans1
df.loc[df['Answer'] == 1, '1'] = sent1

df.loc[df['Answer'] == 2, 'sentence'] = ans2
df.loc[df['Answer'] == 2, '2'] = sent2


df.loc[df['Answer'] == "원인", 'question'] = "결과"
df.loc[df['Answer'] == "임시", 'question'] = "원인"

print(df)

df.to_csv('dataset/task3_COPA/SKT_COPA_Train_flip1.tsv', sep="\t", index = False)