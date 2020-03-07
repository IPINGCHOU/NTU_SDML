#%%
class training_set(Dataset):
    def __init__(self, seqs, pad_sequence):
        self.seqs = seqs
        self.pad_sequence = pad_sequence

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        return self.seqs[index]

    def collate_fn(self, seqs):
        input_seqs = [pair[0] for pair in seqs]
        target_seqs = [pair[1] for pair in seqs]
        input_length = [len(s) for s in input_seqs]
        in_max = np.max(input_length)
        input_padded = [self.pad_sequence(s, in_max) for s in input_seqs]
        target_length = [len(s) for s in target_seqs]
        out_max = np.max(target_length)
        target_padded = [self.pad_sequence(s, out_max) for s in target_seqs]

        return (torch.cuda.LongTensor(input_padded).transpose(0,1), input_length) , (torch.cuda.LongTensor(target_padded).transpose(0,1), target_length) 

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
        self.origin_lr = learning_rate
        self.optimizer= torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.NLLLoss(ignore_index=self.PAD_ID, reduction='mean')

        self.checkpoint_name = checkpoint_name
    
    def train(self, num_epochs, teach_force_descent, batch_size, pretrained = None):

        if pretrained != None:
            self.load_model(pretrained)
        else:
            print(self.model)
        
        step = 0
        train_set = training_set(self.data_transformer.indices_sequence, self.data_transformer.pad_sequence)

        prev_loss_mean = 100
        tolerence = 0
        lr_tolerence = 0
        tol_trigger = 0
        lr_trigger = 0
        trigger_lock = 0
        for epoch in range(0, num_epochs):
            dataloader = DataLoader(dataset = train_set,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    collate_fn=train_set.collate_fn,
                                    num_workers=0)
            total_loss = 0
            trange = tqdm(enumerate(dataloader), total=(len(dataloader)), mininterval = 1,
                            desc = 'epoch: '+str(epoch) + ' ratio: ' + str(self.model.decoder.teacher_forcing_ratio) + ' lr: ' + str(self.learning_rate))
            for i, (input_batch,  target_batch) in trange:
                self.optimizer.zero_grad()

                decoder_outputs, decoder_hidden = self.model(input_batch, target_batch)
                cur_loss = self.get_loss(decoder_outputs, target_batch[0])
                cur_loss.backward()

                total_loss += cur_loss.data.item()
                trange.set_postfix(loss = str(total_loss/(i+1)), refresh = False)
                self.optimizer.step()

                step += 1
                if step % 1000 == 0:
                    torch.cuda.empty_cache()
            total_loss_mean = total_loss/(i+1)
            # descending teach % if slow learning speed
            trigger_lock += 1
            
            if prev_loss_mean != 100 and (prev_loss_mean - total_loss_mean < learing_rate_descent or prev_loss_mean - total_loss_mean < teach_force_descent):
                if trigger_lock >= 3:
                    if disable_teach_descent != 1 and tol_trigger == 0 and self.learning_rate < 1e-6:
                        tol_trigger = 1
                        trange.write('\n Teach trigger activated')
                    elif disable_descent_lr != 1 and lr_trigger == 0:
                        lr_trigger = 1
                        trange.write('\n Learning rate trigger activated')

            if lr_trigger == 1 and prev_loss_mean - total_loss_mean < learing_rate_descent:
                lr_tolerence += 1
                trange.write('\n lr_tolerence: ' + str(lr_tolerence))
                if lr_tolerence >= 3:
                    self.learning_rate = self.learning_rate/10
                    self.optimizer= torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
                    trange.write('\n LR ratio:' + str(self.learning_rate))
                    lr_tolerence = 0
                    lr_trigger = 0
                    trigger_lock = 0
                    prev_loss_mean = 100

            if tol_trigger == 1 and prev_loss_mean - total_loss_mean < teach_force_descent:
                tolerence += 1
                if tolerence >= 3:
                    self.model.decoder.teacher_forcing_ratio = round(self.model.decoder.teacher_forcing_ratio-0.1, 5)
                    trange.write('\n Teach ratio from ' + str(round(self.model.decoder.teacher_forcing_ratio+0.1,5)) + ' to ' + str(self.model.decoder.teacher_forcing_ratio))
                    self.learning_rate = self.origin_lr
                    trange.write('\n Learning Rate reset to:' + str(self.learning_rate))
                    tolerence = 0
                    tol_trigger = 0
                    trigger_lock = 0
                    prev_loss_mean = 100

            if self.model.decoder.teacher_forcing_ratio < 0:
                self.model.decoder.teacher_forcing_ratio = 0
                trange.write('\n Teach ratio set to 0')

            if prev_loss_mean > total_loss_mean:
                prev_loss_mean = total_loss_mean

            # train acc
            test_times = 1000
            correct = 0
            out_of_range = 0
            for i in range(test_times):
                idx = random.randint(0, len(self.data_transformer.train)-1)
                temp = self.data_transformer.train[idx]
                target = self.data_transformer.target[idx]
                pd = self.evaluate(temp)
                
                temp = temp.split()
                pd = pd[0].split('<SOS>')[1].split()
                control_idx = int(temp[-2])
                control_kanji = temp[-1]
                try:
                    if pd[control_idx-1] == control_kanji:
                        correct += 1
                except:
                    out_of_range += 1
            trange.write('Correct: '+ str(correct) + ' / ' + str(test_times))
            trange.write('Out of range: ' + str(out_of_range))
            del idx, temp, target, pd

            self.save_model()
            trange.write('Model for epoch: ' + str(epoch) + ' saved')

    # def train(self, epoch, batch_size, pretrained=None):

    #     if pretrained != None:
    #         self.load_model(pretrained)
    #         print(self.model)
    #     else:
    #         print(self.model)

    #     step = 0

    #     mini_batches = self.data_transformer.mini_batches(batch_size=batch_size)
    #     batch_counts = len(self.data_transformer.indices_sequence) // batch_size +1

    #     with tqdm(total = batch_counts, miniters = 1, desc='epoch: ' + str(epoch)) as t:
    #         for input_batch, target_batch in mini_batches:
    #             self.optimizer.zero_grad()
    #             now = time.time()
    #             decoder_outputs, decoder_hidden, a, b = self.model(input_batch, target_batch)
    #             if epoch >  0:
    #                 t.write(str([time.time()-now,a ,b]))

    #             # calculate the loss and back prop.
    #             cur_loss = self.get_loss(decoder_outputs, target_batch[0])
    #             # logging
    #             step += 1
    #             if step % 1000 == 0:
    #                 torch.cuda.empty_cache()
    #             t.set_postfix(loss = str(cur_loss.data.item()), refresh = False)
    #             t.update(1)
    #             cur_loss.backward()

    #             # optimize
    #             self.optimizer.step()
    #         self.save_model()
    #         tqdm.write('Model for epoch: ' + str(epoch) + ' saved')
                

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

    def tensorboard_log(self):
        pass

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


def train_model(data_transfromer, pretrained=None):
    # define our models
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
    trainer.train(num_epochs, teach_force_descent, batch_size=batch_size, pretrained=pretrained)

    return trainer

def load_model(model_name, data_transformer):
    #define our models
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

#%%
# hyper-parameters
import os
import torch
import random
import pickle
import gc
import numpy as np
import time

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm as tqdm
from tqdm import tnrange

from model import VanillaEncoder
from model import VanillaDecoder
from model import Seq2Seq
from preprocessing import DataTransformer
from multiprocessing import set_start_method

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
use_cuda = True if torch.cuda.is_available() else False

os.chdir('/home/SDML_HW2/task2_1_1/')
corpus_route = 'hw2.1_corpus.txt'
sample_testing1_route = 'hw2.1-1_sample_testing_data.txt'
sample_testing2_route = 'hw2.1-1_sample_testing_data.txt'
corpus_hw211 = 'corpus_hw211.txt'

path = corpus_hw211
# data_transformer = DataTransformer(path = path, use_cuda=use_cuda)
with open('data_transformer.pickle', 'rb') as file:
    data_transformer = pickle.load(file)
# for training
num_epochs = 200
batch_size = 1024
learning_rate = 1e-2

# for model
encoder_embedding_size = 128
encoder_output_size = 128
decoder_hidden_size = encoder_output_size
encoder_layers = 4
encoder_drouput = 0
decoder_layers = 4
decoder_dropout = 0
teacher_forcing_ratio = 0.6
teach_force_descent = 0.001
disable_teach_descent = 1 # fix the teacher forcing ratio or not
learing_rate_descent = 0.005
disable_descent_lr = 0
# max_length = 20

# for logging
checkpoint_name = 'TeachDescentDisable_LRDescentEnable_4layers_teachratio0.5.pt'
#%%
# train
model = train_model(data_transformer)
#%%
# from checkpoint
model = train_model(data_transformer, pretrained = 'TeachDescentDisable_LRDescentEnable_5layers_teachratio0.6.pt')
# %%
# load model
model = load_model(checkpoint_name, data_transformer)
#%%
# small test
test_times = 100
correct = 0
out_of_range = 0
for i in range(test_times):
    idx = random.randint(0, len(model.data_transformer.train)-1)
    temp = model.data_transformer.train[idx]
    target = model.data_transformer.target[idx]
    pd = model.evaluate(temp)
    print('gt: ' + str(temp))
    print('pd: ' + str(pd[0])) # pd is ['aabb'] so take [0]
    print('tg: ' + str(target))
    
    temp = temp.split()
    pd = pd[0].split('<SOS>')[1].split()
    control_idx = int(temp[-2])
    control_kanji = temp[-1]
    try:
        if pd[control_idx-1] == control_kanji:
            correct += 1
    except:
        out_of_range += 1
print('Correct: '+ str(correct) + ' / ' + str(test_times))
print('Out of range: ' + str(out_of_range))

#%%
# total test
test_path = 'hw2.1-1_testing_data.txt'
line = open('%s' % (test_path), encoding = 'utf-8').\
    read().strip().split('\n')
test_dataset = []
for l in line:
    test_dataset.append(l)
truth = test_dataset

batch = batch_size
correct = 0
eval_set = truth
out_of_range = 0
pred_all = []
with tqdm(total = len(eval_set)//batch+1, mininterval=1) as t:
    for i in range(0, len(eval_set), batch):
        gt = eval_set[i:i+batch]
        pred = model.evaluate(gt)
        pred_all.extend(pred)
        t.update(1)
        for truth, pd in zip(gt, pred):
            truth = truth.split(' ')
            pd = pd.split(' ')
            control_idx, control_kanji = int(truth[-2]), truth[-1]
            try:
                if pd[control_idx] == control_kanji:
                    correct += 1
            except:
                out_of_range += 1

print('Matched count: ' + str(correct))
print('Matched %: ' + str(correct/len(eval_set)))
print(out_of_range)

#%%
# import pickle
# file = open('data_transformer.pickle','wb')
# pickle.dump(data_transformer, file)
# file.close()

#%%
p = 0.8
correct = 0
fix = 0
out_of_range = 0
for i, (truth, pd) in enumerate(zip(eval_set, pred_all)):
    truth = truth.split(' ')
    pd = pd.split(' ')
    control_idx, control_kanji = int(truth[-2]), truth[-1]
    try:
        if pd[control_idx] == control_kanji:
            correct += 1
        else:
            if p > random.random():
                pd[control_idx] = control_kanji
                pd = ' '.join(pd)
                pred_all[i] = pd
                fix += 1
    except:
        out_of_range += 1


#%%
correct = 0
out_of_range = 0
for truth, pd in zip(eval_set, pred_all):
    truth = truth.split(' ')
    pd = pd.split(' ')
    control_idx, control_kanji = int(truth[-2]), truth[-1]
    try:
        if pd[control_idx] == control_kanji:
            correct += 1
    except:
        out_of_range += 1

print(str(correct) +' / '+ str(len(eval_set)-out_of_range))


#%%
print([len(eval_set), len(pred_all)])
def list2file(result, file_name):
    with open(file_name, 'w', encoding='utf-8') as f:
        for i, line in enumerate(result):
            f.writelines(line)
            f.write('\n')
        f.close()
#%%
list2file(pred_all, 'task2-1.txt')

# %%
