# deep-bind-modified

This code is a modification of the code by https://github.com/kundajelab/DeepBindToKeras which converts deepbind model to keras. 
The original code worked only for a fixed length DNA sequence and the scores were displayed in a line. 
This code is modified to work with any any sequence length and any number of IDs. The output is in the form of a dataframe for better readability. 
However, when using large data, the execution time is very big and hence its better to use cluster computing. 
