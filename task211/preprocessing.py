#%%
import os
import numpy as np
import torch
from tqdm import tqdm as tqdm
from torch.autograd import Variable

os.chdir('/home/SDML_HW2/task2_1_1/')
# corpus_route = 'hw2.1_corpus.txt'
corpus_hw211 = 'corpus_hw211.txt'
# sample_testing1_route = 'hw2.1-1_sample_testing_data.txt'
# sample_testing2_route = 'hw2.1-1_sample_testing_data.txt'


class Vocabulary(object):
    def __init__(self):
        self.char2idx = {'<SOS>':0, '<EOS>':1, '<PAD>':2, '<UNK>':3}
        self.idx2char = {0 : '<SOS>', 1:'<EOS>', 2:'<PAD>', 3:'<UNK>'}
        self.num_chars = 4
        self.max_length = 0
        self.word_list = []
        self.max_length = 15

        for i in list(range(1,31,1)):
            char = str(i)
            self.char2idx[char] = self.num_chars
            self.idx2char[self.num_chars] = char
            self.num_chars += 1

    def build_vocab(self, corpus_path):
        print('building chinese corpus...')
        with open(corpus_path, 'r', encoding = 'utf-8') as dataset:
            for i, sentence in enumerate(dataset):
                sentence = sentence.split('\n')[0].split()
                for char in sentence:
                    if char not in self.char2idx:
                        self.char2idx[char] = self.num_chars
                        self.idx2char[self.num_chars] = char
                        self.num_chars += 1
        print('done')
        print('Sentence count: ' + str(i))
        print('Kanji count: ' + str(len(self.char2idx)))

    def sequence_to_indices(self, sequence, add_eos=False, add_sos=False):
        index_sequence = [self.char2idx['<SOS>']] if add_sos else []

        for char in sequence.split('\n')[0].split():
            if char not in self.char2idx:
                index_sequence.append(self.char2idx['<UNK>'])
            else:
                index_sequence.append(self.char2idx[char])
        
        if add_eos:
            index_sequence.append(self.char2idx['<EOS>'])
        
        return index_sequence

    def indices_to_sequence(self, indices):
        sequence = ''
        for idx in indices:
            char = self.idx2char[int(idx)]
            if char == '<EOS>':
                sequence += char
                break
            else:
                sequence += char + ' ' 
        return sequence
    
    def __str__(self):
        str = "Vocab information:\n"
        for idx, char in self.idx2char.items():
            str += "Char: %s Index: %d\n" % (char, idx)
        return str

class DataTransformer(object):
    
    def __init__(self, path, use_cuda):
        self.indices_sequence = []
        self.use_cuda = use_cuda
        self.train = []
        self.target = []
        
        # Load and build vocab
        self.vocab = Vocabulary()
        self.vocab.build_vocab(path)
        self.PAD_ID = self.vocab.char2idx['<PAD>']
        self.SOS_ID = self.vocab.char2idx['<SOS>']
        self.vocab_size = self.vocab.num_chars
        self.max_length = self.vocab.max_length

        self._build_training_set(path)
    
    def read_data(self, file_name, with_split=False):
        line = open('%s' % (file_name), encoding = 'utf-8').\
        read().strip().split('\n')
        output = []
        if with_split:
            for l in line:
                l=[char for char in l]
                output.append(l)
        else:
            for l in line:
                output.append(l)

        return output

    def _build_training_set(self, path):
        print('building training set...')
        trainingset = self.read_data(file_name = path)
        for i, now_sentence in enumerate(trainingset):
            try:
                target_sentence = trainingset[i+1]
            except:
                break
            self.train.append(now_sentence.split('\n')[0])
            self.target.append(target_sentence.split('\n')[0])
            now_indices_seq = self.vocab.sequence_to_indices(now_sentence, add_eos = False)
            target_indices_seq = self.vocab.sequence_to_indices(target_sentence, add_eos = False)[:-2]
            self.indices_sequence.append([now_indices_seq, target_indices_seq])

        # for i, sentence in enumerate(trainingset):
        #     self.train.append(sentence.split('\n')[0])
        #     indices_seq = self.vocab.sequence_to_indices(sentence, add_eos = False)
        #     # make input and target the same for auto-encoder
        #     self.indices_sequence.append([indices_seq, indices_seq])
        print('training set builded')
        print('Counts : ' + str(i))
        
    def mini_batches(self, batch_size):
        input_batches = []
        target_batches = []
        
        np.random.shuffle(self.indices_sequence)
        mini_batches = [
            self.indices_sequence[k : k + batch_size]
            for k in range(0, len(self.indices_sequence), batch_size)
        ]
        for batch in mini_batches:
            input_seqs = [pair[0] for pair in batch]
            target_seqs = [pair[1] for pair in batch]
            input_length = [len(s) for s in input_seqs]
            in_max = np.max(input_length)
            input_padded = [self.pad_sequence(s, in_max) for s in input_seqs]
            target_length = [len(s) for s in target_seqs]
            out_max = np.max(target_length)
            target_padded = [self.pad_sequence(s, out_max) for s in target_seqs]

            # time * batch
            input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
            target_var = Variable(torch.LongTensor(target_padded)).transpose(0 , 1)
            
            if self.use_cuda:
                input_var = input_var.cuda()
                target_var = target_var.cuda()
            
            yield (input_var, input_length) , (target_var, target_length)
   
    def pad_sequence(self, sequence, max_length):
        sequence += [self.PAD_ID for i in range(max_length - len(sequence))]
        return sequence
    
    def evaluation_batch(self, words):
        evaluation_batch = []

        for word in words:
            indices_seq = self.vocab.sequence_to_indices(word, add_eos = False)
            evaluation_batch.append(indices_seq)
        
        # seq_pairs = sorted(evaluation_batch, key = lambda seqs:len(seqs[0]), reverse = True)
        input_seqs = evaluation_batch
        input_length = [len(s) for s in input_seqs]
        in_max = np.max(input_length)
        input_padded = [self.pad_sequence(s, in_max) for s in input_seqs]
        input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
        if self.use_cuda:
            input_var = input_var.cuda()
        
        return input_var, input_length

#%%
# if __name__ == '__main__':
    # vocab = Vocabulary()
    # vocab.build_vocab(corpus_hw211)

    # test = str('侶 谆 钿')
    # print("Sequence before transformed:", test)
    # ids = vocab.sequence_to_indices(test, test_vocab= False)
    # print("Indices sequence:", ids)
    # sent = vocab.indices_to_sequence(ids)
    # print("Sequence after transformed:",sent)

    # data_transformer = DataTransformer(path = corpus_hw211, use_cuda=False)
    # for ib, tb in data_transformer.mini_batches(batch_size=3):
    #     print("B0-0")
    #     print(ib, tb)
    #     break


# %%
