#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import keras
import tensorflow as tf

def convert_model(file_name, input_len, window_size=0):
    
    data = [x.rstrip() for x in open(file_name).read().split("\n")]
    
    reverse_complement = int(data[1].split(" = ")[1])
    num_detectors = int(data[2].split(" = ")[1])
    detector_len = int(data[3].split(" = ")[1])
    has_avg_pooling = int(data[4].split(" = ")[1])
    num_hidden = int(data[5].split(" = ")[1])
    
    if (window_size < 1):
        window_size = (int)(detector_len*1.5) #copying deepbind code
    if (window_size > input_len):
        window_size = input_len

    detectors = (np.array(
        [float(x) for x in data[6].split(" = ")[1].split(",")])
        .reshape(detector_len, 4, num_detectors))
    biases = np.array([float(x) for x in data[7].split(" = ")[1].split(",")])
    weights1 = np.array([float(x) for x in data[8].split(" = ")[1].split(",")]).reshape(
                        num_detectors*(2 if has_avg_pooling else 1),
                        (1 if num_hidden==0 else num_hidden))
    if (has_avg_pooling > 0):
        #in the orignal deepbind model, these weights are interleaved.
        #what a nightmare.
        weights1 = weights1.reshape((num_detectors,2,-1))
        new_weights1 = np.zeros((2*num_detectors, weights1.shape[-1]))
        new_weights1[:num_detectors, :] = weights1[:,0,:]
        new_weights1[num_detectors:, :] = weights1[:,1,:]
        weights1 = new_weights1
    biases1 = np.array([float(x) for x in data[9].split(" = ")[1].split(",")]).reshape(
                        (1 if num_hidden==0 else num_hidden))
    if (num_hidden > 0):
        #print("Model has a hidden layer")
        weights2 = np.array([float(x) for x in data[10].split(" = ")[1].split(",")]).reshape(
                        num_hidden,1)
        biases2 = np.array([float(x) for x in data[11].split(" = ")[1].split(",")]).reshape(
                        1)
    
    
    def seq_padding(x):
        return tf.pad(x,
                [[0, 0],
                 [detector_len-1, detector_len-1],
                 [0, 0]],
                mode='CONSTANT',
                name=None,
                constant_values=0.25)

    
    input_tensor = keras.layers.Input(shape=(None,4))
    padding_out_fwd = keras.layers.Lambda(seq_padding)(input_tensor)
    conv_layer = keras.layers.Conv1D(filters=num_detectors,
                                  kernel_size=detector_len,
                                  activation="relu")
    conv_out_fwd = conv_layer(padding_out_fwd)
    pool_out_fwd = keras.layers.MaxPooling1D(pool_size=(window_size+detector_len-1),
                                             strides=1)(conv_out_fwd)
    if (has_avg_pooling > 0):
        #print("Model has average pooling")
        gap_out_fwd = keras.layers.AveragePooling1D(pool_size=(window_size+detector_len-1),
                                                     strides=1)(conv_out_fwd)
        pool_out_fwd = keras.layers.Concatenate(axis=-1)([pool_out_fwd, gap_out_fwd])        
    dense1_layer = keras.layers.Dense((1 if num_hidden==0 else num_hidden))
    dense1_out_fwd = keras.layers.TimeDistributed(dense1_layer)(pool_out_fwd)
    if (num_hidden > 0):
        dense1_out_fwd = keras.layers.Activation("relu")(dense1_out_fwd)
        dense2_layer = keras.layers.Dense(1)
        dense2_out_fwd = keras.layers.TimeDistributed(dense2_layer)(dense1_out_fwd)
    
    if (reverse_complement > 0):
        #print("Model has reverse complementation")
        padding_out_rev = keras.layers.Lambda(lambda x: x[:,::-1,::-1])(padding_out_fwd)
        conv_out_rev = conv_layer(padding_out_rev)
        pool_out_rev = keras.layers.MaxPooling1D(pool_size=(window_size+detector_len-1),
                                             strides=1)(conv_out_rev)
        if (has_avg_pooling > 0):
            #print("Model has average pooling")
            gap_out_rev = keras.layers.AveragePooling1D(pool_size=(window_size+detector_len-1),
                                                     strides=1)(conv_out_rev)
            pool_out_rev = keras.layers.Concatenate(axis=-1)([pool_out_rev, gap_out_rev])
        dense1_out_rev = keras.layers.TimeDistributed(dense1_layer)(pool_out_rev)
        if (num_hidden > 0):
            dense1_out_rev = keras.layers.Activation("relu")(dense1_out_rev)
            dense2_out_rev = keras.layers.TimeDistributed(dense2_layer)(dense1_out_rev)
    
    cross_seq_max = keras.layers.Lambda(lambda x: tf.reduce_max(x,axis=1)[:,0],
                                        output_shape=lambda x: (None,1))
    
    if (reverse_complement > 0):
        if (num_hidden > 0):
            max_fwd = cross_seq_max(dense2_out_fwd)
            max_rev = cross_seq_max(dense2_out_rev)
            output = keras.layers.Maximum()([max_fwd, max_rev])
        else:
            max_fwd = cross_seq_max(dense1_out_fwd)
            max_rev = cross_seq_max(dense1_out_rev)
            output = keras.layers.Maximum()([max_fwd, max_rev])
    else:
        if (num_hidden > 0):
            output = cross_seq_max(dense2_out_fwd)
        else:
            output = cross_seq_max(dense1_out_fwd)
        
    #print(input_tensor)   
    model = keras.models.Model(inputs = [input_tensor],
                               outputs = [output])
    model.compile(loss="mse", optimizer="adam")
    conv_layer.set_weights([detectors, biases])
    dense1_layer.set_weights([weights1, biases1])
    if (num_hidden > 0):
        dense2_layer.set_weights([weights2, biases2])
        
    return model

def onehot_encode_sequences(sequences):
    onehot = []
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3, 'u': 3}
    for sequence in sequences:
        arr = np.zeros((len(sequence), 4)).astype("float")
        for (i, letter) in enumerate(sequence):
            arr[i, mapping[letter]] = 1.0
        onehot.append(arr)
    return onehot


# In[2]:


import glob
import os
import re 

import sys
dna = []
fasta = []
test = []

####Reading sequences from FASTA txt files and converting them into a list 

### Change path here

file_list = glob.glob(os.path.join(os.getcwd(), "./la", "*.fa"))

new = []
dna = []
sequences = []

for file_path in file_list:
    with open(file_path) as file_one:
        for line in file_one:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                active_sequence_name = line[1:]
                dna.append(line)
                if active_sequence_name not in fasta:
                    test.append(''.join(fasta))
                    fasta = []
                continue
            sequence = line
            fasta.append(sequence)

### Flush the last fasta block to the test list
if fasta:
    test.append(''.join(fasta))
    


#print(test)
#print(dna)

# In[4]:

### Removing any empty element from list 

new_sequences= []
for string in test:
     if (string != ""):
            new_sequences.append(string)
            

# In[8]:



###Restricting the sequence to only 20 characters with sliding window = 3
n = 20
a=[]
b =[]
list1 =[]
list2 = []
final_seq = []    
for i in new_sequences:
    if len(i) > 20:
        a.append([i[j-n:j] for j in range(n, len(i)+n, 3)])
   
    
    else:
        final_seq.append([i])

#final_seq.append(list1)        
#print (final_seq)
#print(list1)
for i in a:
    for j in i:
        if len(j) >=20:
            list2.append(j)
    list1.append(list2)
    list2 = []

### Final output is a list of list 

final_seq.extend(list1)
#print(final_seq)
#print(list1)




# In[9]:

### Creating new data frame for output 

import pandas as pd
df = pd.DataFrame([])


# In[10]:


new_l = []
new_seq = []
ID = []
new_position = []
new_score =[]


# In[11]:


import itertools
count = 0 
#yo = 0 
#### Get the ID files from local directory - change path here 
### Only few ID files are taken for testing sake. This part takes a little longer to run when more ID files are inputed. 
### This code works for all sequences length, it splits the sequence into 20 and slides by a window size of 3

### Change path
path = "./minidb"   
for file_name in glob.glob(os.path.join(path, '*.txt')):   
    print(count)
    
    ## getting only the filename without extension
    tail = os.path.basename(file_name)[0:10]

    
    for i in final_seq:
        onehot_sequences = onehot_encode_sequences(i)
                    
    ## Getting the scores and saving it into a dataframe   
    for i,a in zip(final_seq,dna):

        
        l = 1
        for y,j in enumerate(i):
       
            new_l.append(a)
            new_seq.append(j)
            ID.append(tail)
       
            model = convert_model(file_name = file_name, input_len= len(j))
            l = l+3
            
        l = 1     
        for k in range(len(i)):
            
            new_position.append(l)
            

            for x in model.predict(np.array(onehot_sequences[k:k+1])[:,:,:]):
                new_score.append(str(x))
            
            l = l+3
    
    

    count = count + 1
### Writing into data frame
df['Position']= new_position
df['Score']= new_score
df['Seq'] = new_seq
df['ID'] = ID
df['Type']= new_l

# In[12]:


### Mapping IDs with corresponding protein name
dictionary = {}

with open("matrix2proteinname.map", "r") as file:
    for line in file:
        key, value = line.strip().split("\t")
        dictionary[key] = value
print(dictionary)


# In[13]:


df["Protein"] = df["ID"].map(dictionary) 



# In[12]:

### output to csv file

df.to_csv('output.csv', index = False)






