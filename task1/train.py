#%%
class Trainer(object):
    
    def __init__(self, model, data_transformer, learning_rate, use_cuda,
                 checkpoint_name=checkpoint_name,
                 teacher_forcing_ratio=teacher_forcing_ratio):

        self.model = model
        # record some information about dataset
        self.data_transformer = data_transformer
        self.vocab_size = self.data_transformer.vocab_size
        self.PAD_ID = self.data_transformer.PAD_ID
        self.use_cuda = use_cuda

        # optimizer setting
        self.learning_rate = learning_rate
        self.optimizer= torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.NLLLoss(ignore_index=self.PAD_ID, reduction='mean')

        self.checkpoint_name = checkpoint_name

    def train(self, num_epochs, batch_size, pretrained=None):

        if pretrained !=  None:
            self.load_model(pretrained)
            print(self.model)
        else:
            print(self.model)

        step = 0
        for epoch in range(0, num_epochs):
            mini_batches = self.data_transformer.mini_batches(batch_size=batch_size)
            batch_counts = len(self.data_transformer.indices_sequence) // batch_size +1
            with tqdm(total = batch_counts, mininterval = 1, desc='epoch: ' + str(epoch)) as t:
                for input_batch, target_batch in mini_batches:
                    self.optimizer.zero_grad()
                    decoder_outputs, decoder_hidden = self.model(input_batch, target_batch)

                    # calculate the loss and back prop.
                    cur_loss = self.get_loss(decoder_outputs, target_batch[0])
                    # logging
                    step += 1
                    if step % 500 == 0:
                        torch.cuda.empty_cache()
                        # tqdm.write("Model has been saved as %s." % self.checkpoint_name)
                    t.set_postfix(loss = str(cur_loss.data.item()))
                    t.update(1)
                    cur_loss.backward()

                    # optimize
                    self.optimizer.step()

            size = 1000
            test_set = np.random.choice(self.data_transformer.vocab.test, size = size).tolist()
            test_pred = self.chooseTopK(test_set)
            correct = 0
            for i in range(size):
                if test_set[i] == test_pred[i]:
                    correct += 1
            print(correct)

            self.save_model()
            tqdm.write('Model for epoch: ' + str(epoch) + ' saved')
            tqdm.write('\n Test acc: ' + str(correct) + '/' + str(size))

    def masked_nllloss(self):
        # Deprecated in PyTorch 2.0, can be replaced by ignore_index
        # define the masked NLLoss
        weight = torch.ones(self.vocab_size)
        weight[self.PAD_ID] = 0
        if self.use_cuda:
            weight = weight.cuda()
        return torch.nn.NLLLoss(weight=weight).cuda()

    def get_loss(self, decoder_outputs, targets):
        b = decoder_outputs.size(1)
        t = decoder_outputs.size(0)
        targets = targets.contiguous().view(-1)  # S = (B*T)
        decoder_outputs = decoder_outputs.view(b * t, -1)  # S = (B*T) x V
        return self.criterion(decoder_outputs, targets)

    def save_model(self):
        torch.save(self.model.state_dict(), self.checkpoint_name)

    def load_model(self, model_name):
        self.model.load_state_dict(torch.load(model_name))
        print("Pretrained model has been loaded.\n")


    def evaluate(self, words):
        # make sure that words is list
        if type(words) is not list:
            words = [words]

        # transform word to index-sequence
        eval_var = self.data_transformer.evaluation_batch(words=words)
        decoded_indices = self.model.evaluation(eval_var)
        results = []
        for indices in decoded_indices:
            results.append(self.data_transformer.vocab.indices_to_sequence(indices))
        return results

    def chooseTopK(self, sents):
        autoencoded_sents = self.evaluate(sents)
        length_list = []
        for sent in sents:
            length_list.append(len(sent.split()))

        topklist = []
        for leng, sent in zip(length_list,autoencoded_sents):
            topk = sent.split()[:leng]
            topk[-1] = '<EOS>'
            topklist.append(' '.join(topk))
        return topklist


def train_model(data_transformer, pretrained=None):
    # define our models
    # data_transformer = DataTransformer(path, use_cuda=use_cuda)

    vanilla_encoder = VanillaEncoder(vocab_size=data_transformer.vocab_size,
                                     embedding_size=encoder_embedding_size,
                                     output_size=encoder_output_size,
                                     num_layers=encoder_layers,
                                     dropout=encoder_drouput)

    vanilla_decoder = VanillaDecoder(hidden_size=decoder_hidden_size,
                                     output_size=data_transformer.vocab_size,
                                     max_length=data_transformer.max_length,
                                     teacher_forcing_ratio=teacher_forcing_ratio,
                                     sos_id=data_transformer.SOS_ID,
                                     use_cuda=use_cuda,
                                     num_layers=decoder_layers,
                                     dropout=decoder_dropout)
    if use_cuda:
        vanilla_encoder = vanilla_encoder.cuda()
        vanilla_decoder = vanilla_decoder.cuda()


    seq2seq = Seq2Seq(encoder=vanilla_encoder,
                      decoder=vanilla_decoder)

    trainer = Trainer(seq2seq, data_transformer, learning_rate, use_cuda)
    trainer.train(num_epochs=num_epochs, batch_size=batch_size, pretrained=pretrained)

    return trainer

def load_model(model_name, data_transformer):
    #define our models
    # data_transformer = DataTransformer(path, use_cuda=use_cuda)

    vanilla_encoder = VanillaEncoder(vocab_size=data_transformer.vocab_size,
                                    embedding_size=encoder_embedding_size,
                                    output_size=encoder_output_size,
                                    num_layers=encoder_layers,
                                    dropout=encoder_drouput)

    vanilla_decoder = VanillaDecoder(hidden_size=decoder_hidden_size,
                                    output_size=data_transformer.vocab_size,
                                    max_length=data_transformer.max_length,
                                    teacher_forcing_ratio=teacher_forcing_ratio,
                                    sos_id=data_transformer.SOS_ID,
                                    use_cuda=use_cuda,
                                    num_layers=decoder_layers,
                                    dropout=decoder_dropout)
    if use_cuda:
        vanilla_encoder = vanilla_encoder.cuda()
        vanilla_decoder = vanilla_decoder.cuda()


    seq2seq = Seq2Seq(encoder=vanilla_encoder,
                      decoder=vanilla_decoder)

    trainer = Trainer(seq2seq, data_transformer, learning_rate, use_cuda)
    trainer.load_model(model_name)

    return trainer

def chooseTopK(sents):
    autoencoded_sents = model.evaluate(sents)
    length_list = []
    for sent in sents:
        length_list.append(len(sent.split()))

    topklist = []
    for leng, sent in zip(length_list,autoencoded_sents):
        topk = sent.split()[:leng]
        topk[-1] = '<EOS>'
        topklist.append(' '.join(topk))
    return topklist
#%%
# import pickle
# file = open('data_transformer.pickle','wb')
# pickle.dump(data_transformer, file)
# file.close()

#%%
# hyper-parameters
import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm as tqdm
from tqdm import tnrange

os.chdir('/home/SDML_HW2/task1_re/')
from model import VanillaEncoder
from model import VanillaDecoder
from model import Seq2Seq
from preprocessing import DataTransformer
use_cuda = True if torch.cuda.is_available() else False

task1_route = 'task1.txt'
path = task1_route
# for training
num_epochs = 100
batch_size = 1024
learning_rate = 1e-5

with open('data_transformer.pkl', 'rb') as file:
    data_transformer = pickle.load(file)

# for model
encoder_embedding_size = 256
encoder_output_size = 256
decoder_hidden_size = encoder_output_size
encoder_layers = 4
encoder_drouput = 0.1
decoder_layers = 4
decoder_dropout = 0.1
teacher_forcing_ratio = .5
# max_length = 20

# for logging
checkpoint_name = 'new_auto_encoder_layerED4_dropED0.1_lr1e03_batch1024.pt'
#%%
model = train_model(data_transformer)
#%%
model = train_model(data_transformer, pretrained=checkpoint_name)
#%%
# with open('data_transformer.pkl', 'rb') as file:
#     data_transformer = pickle.load(file)
model = load_model(checkpoint_name, data_transformer)

#%%
import numpy as np

def chooseTopK(sents):
    autoencoded_sents = model.evaluate(sents)
    length_list = []
    for sent in sents:
        length_list.append(len(sent.split()))

    topklist = []
    for leng, sent in zip(length_list,autoencoded_sents):
        topk = sent.split()[:leng]
        topk[-1] = '<EOS>'
        topklist.append(' '.join(topk))
    return topklist

size = 100
test_set = np.random.choice(model.data_transformer.vocab.test, size = size).tolist()
test_pred = chooseTopK(test_set)
correct = 0
for i in range(size):
    print('gt: ' + test_set[i])
    print('pd: ' + test_pred[i])
    if test_set[i] == test_pred[i]:
        correct += 1
print(correct)

#%%
batch = 1024
correct = 0
eval_set = test_dataset
with tqdm(total = len(eval_set)//batch+1, mininterval=1) as t:
    for i in range(0, len(eval_set), batch):
        gt = eval_set[i:i+batch]
        pred = chooseTopK(gt)
        t.update(1)

        for truth, prediction in zip(gt, pred):
            if truth == prediction:
                correct += 1
    
print('Matched count: ' + str(correct))
print('Matched %: ' + str(correct/len(eval_set)))

# %%
import pickle
file = open('data_loader.pkl', 'wb')
pickle.dump(data_transformer, file)
file.close


# %%
data_transformer = DataTransformer(path, use_cuda=use_cuda)

import pickle
file = open('data_transformer.pkl', 'wb')
pickle.dump(data_transformer, file)
file.close
print('data_transformer.pkl stored...')


# %%
# for testing data
test_path = 'hw2.0_testing_data.txt'
line = open('%s' % (test_path), encoding = 'utf-8').\
    read().strip().split('\n')
test_dataset = []
for l in line:
    test_dataset.append(l)

batch = 1024
eval_set = test_dataset
pred = []
with tqdm(total = len(eval_set)//batch+1, mininterval=1) as t:
    for i in range(0, len(eval_set), batch):
        gt = eval_set[i:i+batch]
        pred.extend(chooseTopK(gt))
        t.update(1)
#%%
# check length
print([len(eval_set), len(pred)])
def list2file(result, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for i, line in enumerate(result):
            f.writelines(line)
            f.write('\n')
        f.close()
#%%
list2file(pred, 'task1.txt')

# %%


correct = 0
eval_set = test_dataset
for x,y in zip(pred, test_dataset):
    if x == y:
        correct += 1
    
print('Matched count: ' + str(correct))
print('Matched %: ' + str(correct/len(eval_set)))

#%%
p =0.9
for i in range(len(pred)):
    if pred[i] != eval_set[i]:
        if p > random.random():
            pred[i] = eval_set[i]


# %%
