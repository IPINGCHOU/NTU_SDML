# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../../tmp'))
	print(os.getcwd())
except:
	pass
# %%
from tqdm import tqdm_notebook as tqdm
from tqdm import trange


# %%
import random

TASK = 'task2.1-2'
raw_data_filename = 'hw2.1_corpus.txt'
output_filename = TASK + '_corpus.txt'

if TASK[-1] == '1':
    MAXIMUM_NUM_SIGNAL = 1
else:
    MAXIMUM_NUM_SIGNAL = 2
    
trainset = []

def generate_control_signal(next_line, num_signal=1):
    
    random.seed(next_line) 
    control_signals = []
    seqences = range(0, len(next_line))
    if num_signal > len(next_line):
        num_signal = 1
    positions = random.sample(seqences, k=num_signal)
    
    for pos in positions:
        word = next_line[pos]
        
        control_signals.append(str(pos+1)) # Cause next line will be added <SOS>.
        control_signals.append(word)
        
    return control_signals   

with open(raw_data_filename, 'r') as f:
    
    lines = f.read().strip().split('\n')[:-1]
    
    trange = tqdm(enumerate(lines), total=len(lines), desc='Preprocessing Data')
    
    for i, (row) in enumerate(trange):
        
        line = row[1]
        
        # Split sentence into per character list and add <SOS>, <EOS>.
        splitted_line = [ch for ch in line]
        output_line = ['<SOS>']
        output_line.extend(splitted_line)
        output_line.append('<EOS>')
        
        # Append control signal (one)
        control_signal_pairs = []
        
        if i < len(lines) - 1:
            rand_num_signal = random.randint(1, MAXIMUM_NUM_SIGNAL)
            control_signal_pairs = generate_control_signal(lines[i+1], num_signal=rand_num_signal)
        
        if len(control_signal_pairs) > 0:
            output_line.extend(control_signal_pairs)
        
        trainset.append(output_line)
    
    f.close()
        


# %%
with open(output_filename, 'w', encoding='utf-8') as outfile:
    
    trange = tqdm(enumerate(trainset), total=len(trainset), desc='WriteBack')
    
    for j, (_row) in enumerate(trange):
        result = ' '.join(_row[1]).strip()
        result = result + '\n'
        outfile.writelines(result)
    
    outfile.close()


# %%


