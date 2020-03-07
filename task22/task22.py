#%%
import numpy as np
import torch
import pickle
import os
import random
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc_params
from matplotlib.lines import Line2D
os.chdir('/home/SDML_HW2/task22')

#%%
file = open('evaluate_tstep/X.0.pkl', 'rb')
X = pickle.load(file)
file.close()
file = open('dicitonary.pkl', 'rb')
dictionary = pickle.load(file)
file.close()
file = open('embedding.pkl', 'rb')
embedder = pickle.load(file)
file.close()

# %%
def get_kanji(x):
    kanji_list = []
    for val in x:
        kanji_list.append(embedder.index_dict.get(val))
    
    return kanji_list
test = get_kanji(X[:,1])
print(test)

# %%
# abs sum plot
steps = 28
fig = plt.figure()
ims = []
for i in range(0,steps,1):
    file = open('evaluate_tstep/decoderRNN_output1-'+str(i)+'.pkl', 'rb')
    cur_weight = pickle.load(file)
    # print(np.shape(cur_weight.sum(axis=0).sum(axis=0)))
    file.close()

    im = plt.bar(np.arange(128), np.abs(cur_weight).sum(axis = 0), color = 'white')
    ims.append(im)
    
ani = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=1000)
ani.save("test2.gif",writer='pillow')

'''
found neuron number 21 and 91 might handle the control signal
'''
# %%
# var plot
steps = 28
fig = plt.figure()
ims = []
for i in range(0,steps,1):
    file = open('evaluate_tstep/decoderRNN_output0-'+str(i)+'.pkl', 'rb')
    cur_weight = pickle.load(file)
    # print(np.shape(cur_weight.sum(axis=0).sum(axis=0)))
    file.close()

    im = plt.bar(np.arange(128), cur_weight.var(axis = 0), color = 'white')
    ims.append(im)
    
ani = animation.ArtistAnimation(fig, ims, interval=500, repeat_delay=1000)
ani.save("decoderRNNoutput_var.gif",writer='pillow')

'''
found neuron number 60, 62, 47
'''

# %%
'''
take all the control signal out
'''
control_pos = []
for i in range(1024):
    cur_kanji = get_kanji(X[:,i])
    for j in range(len(cur_kanji)):
        try:
            control_pos.append(int(cur_kanji[j]))
        except:
            continue
print(control_pos)

import colorsys
N = np.max(control_pos)
HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
RGB_tuples = list(map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples))
color_plate = []
for i in range(1024):
    color_plate.append(RGB_tuples[control_pos[i]-1])

# %%
steps = 15
row = 5
col = 3
legend_elements = []
plt.rcParams.update({'font.size': 13})
for i in range(N):
    legend_elements.append(Line2D([0], [0], marker='o', color=RGB_tuples[i],
     label='Signal '+str(i+1), markersize=12))


plt.figure(figsize=(30, 18), dpi=80)
for i in range(0,steps,1):
    file = open('evaluate_tstep/decoderRNN_hidden0-'+str(i)+'.pkl', 'rb')
    cur_hidden = pickle.load(file)
    # print(np.shape(cur_weight.sum(axis=0).sum(axis=0)))
    file.close()
    x = cur_hidden[3,:,21]
    y = cur_hidden[3,:,91]
    plt.subplot(row,col,i+1)
    plt.scatter(x,y, color = color_plate, s= 100)
    plt.title('step ' + str(i))
plt.legend(handles = legend_elements, loc = 1,bbox_to_anchor=(1.3,5.5),labelspacing=2)
plt.xlabel("neuron 21")
plt.ylabel("neuron 91")
plt.tight_layout()
# plt.savefig('neuron21_91_step15.png')
plt.show()
# %%
step_add = []
steps = 15
plt.figure(figsize=(10, 6), dpi=80)
legend_elements = []
for i in range(N):
    legend_elements.append(Line2D([0], [0], marker='+', color=RGB_tuples[i],
     label='Signal '+str(i+1)))

for i in range(0,steps,1):
    file = open('evaluate_tstep/decoderRNN_hidden0-'+str(i)+'.pkl', 'rb')
    cur_hidden = pickle.load(file)
    # print(np.shape(cur_weight.sum(axis=0).sum(axis=0)))
    file.close()
    neuron = cur_hidden[3,:,47]
    temp_step = []
    for j in range(N):
        temp_step.append(np.average(neuron[np.where(np.array(control_pos)==(j+1))]))
    step_add.append(temp_step)
step_add = np.array(step_add)

for m in range(N):
    plt.plot(range(0,15,1),step_add[:,m], marker = '+', color = RGB_tuples[m])
plt.legend(handles = legend_elements, loc = 1,bbox_to_anchor=(1.3,1),labelspacing=0.8)
plt.tight_layout()
plt.savefig('neuron47_step15.png')

# %%