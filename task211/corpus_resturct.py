import os
import random
from chinese import ChineseAnalyzer
analzyer = ChineseAnalyzer()
os.chdir('/home/SDML_HW2/task2_1_1/')
corpus_route = 'hw2.1_corpus.txt'
sample_testing1_route = 'hw2.1-1_sample_testing_data.txt'
sample_testing2_route = 'hw2.1-1_sample_testing_data.txt'

def read_data(file_name, cut, with_split=False):
    del_count = 0
    line = open('%s' % (file_name), encoding = 'utf-8').\
        read().strip().split('\n')
    output = []
    if with_split:
        for l in line:
            l=[char for char in l]
            output.append(l)
    else:
        for l in line:
            if len(l) >= cut:
                continue
            output.append(l)

    return output

def list2file(result, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for i, line in enumerate(result):
            f.writelines(line)
        f.close()

def transfer_to_hw211(corpus, cut=25):
    print('reading corpus...')
    corpus_file = read_data(corpus, cut)
    output_file = []
    for i, now_sentence in enumerate(corpus_file):
        output = []
        try:
            next_sentence = corpus_file[i+1]
        except:
            return output_file
        output.append('<SOS>')
        output.extend(now_sentence)
        output.append('<EOS>')
        condition_index = random.randint(0,len(next_sentence)-1)
        condition_kanji = next_sentence[condition_index]
        output.append(str(condition_index+1))
        output.append(condition_kanji)
        output.append('\n')
        output_file.append(' '.join(output))
    print('done')
    return output_file

cut = 25
corpus_hw211 = transfer_to_hw211(corpus_route, cut)
list2file(corpus_hw211, 'corpus_hw211.txt')


