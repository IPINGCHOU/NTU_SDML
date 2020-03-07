
import os
import numpy as np
import torch
from tqdm import tqdm as tqdm
from torch.autograd import Variable

os.chdir('/home/SDML_HW2/task2_1_1/')
corpus_hw211 = 'corpus_hw211.txt'

count = 0
with open(corpus_hw211, 'r', encoding = 'utf-8') as dataset:
    for i, sentence in enumerate(dataset):
        sentence = sentence.split('\n')[0].split()
        if len(sentence) >= 30:
            count += 1

print(count)

#%%
import random
print(random.random())

# %%
