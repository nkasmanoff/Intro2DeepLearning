"""
Using the tflearn wrapper/API, 
I will construct a neural network to analyze the sentiment
of different movie reviews, either
 positive or negative, and use this approach in order to classify future reviews. 
This is the program that is walked through, and in order to see the independently designed software, 
try the other .py file. 

"""


import tflearn # "easiest way to start deep learning."
from tflearn.data_utils import to_categorical, pad_sequences #helper functions

from tflearn.datasets import imdb #a preprocessed data set. Something you don't see in nature! 

# IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                valid_portion=0.1)  #load as a pkl file, a byte oriented object that can easily be converted to lists and tuples later on, or in this code!
				#n words is how many total words we want to look at, and validation corresponds to the size of the validation data to be placed inside, or.1 (test = .1*train in this case.)


#split features and labels. 
trainX, trainY = train
testX, testY = test 

# Data preprocessing
# Sequence padding

trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)

"""Pad sequences? What is this? 
You can't just feed text strings into a neural network. 

This is how these inputs are vectorized. 

This function converts each review into a matrix, and "pads" it. 
Padding is used to ensure consistency in the dimesionality, so for this particular case each
review if the review isn't 100 words, this will pad the vector with zeros until it reaches the length 100. 

I don't totally understand this word2vec proccess, so I will cover it in another section of this repo. 
"""

# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)
#basically OHE. 

# Network building

"""Another confusing part. A normal FFNN will not suffice for this problem, so a
recurrent neural network is employed in order to capture the sequential order of text (Like in a movie review!)
and is programmed below. An important piece of a RNN is the LSTM which helps "remember" the sequence. 

Will be further covered....
"""

net = tflearn.input_data([None, 100]) #input layer, [batch_size, size of input]
net = tflearn.embedding(net, input_dim=10000, output_dim=128)  #we use the previous layers output as the next layers input. 
# in an embedding, words are represented by dense vectors where 
#a vector represents the projection of the word into a continuous
#vector space. There are 10k words in these reviews, so that is how many dim are needed. 

#The position of a word in the learned vector space is referred to as its embedding.

# a word embedding can be learned as part of a deep learning model. 
#This can be a slower approach, but tailors the model to a specific
#training dataset.


net = tflearn.lstm(net, 128, dropout=0.8)
"""Quick summary: 

Every review is padded so that it has 100 dimensions, and the words in the each review (there are 10k total words)
will slowly be re-evaluated by this Neural net as either positive or negative, using deep learning. 

The sequential nature of the review is accounted for using the lstm, which accounts for each word in the sequence (review).

After a ton of training, the vector space of all of these words will be properly defined, such that words like "awesome movie!" corresponds to a good movie,
 and if this model is sophisticated enough "This movie was so good I wanted to cut my eyes out", a sarcastic review, might even be able to get caught. 

 If someone is reading this and knows better please show me!
"""


#and now some housekeeping. This finalizes the model, and confirms all the layers are connected. 2 is the output? 
net = tflearn.fully_connected(net, 2, activation='softmax') #connects the layers. not just adjascent neurons. 
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,#applies a regression to the input. 
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=32)

###Now my stuff!
#Testing

score = model.score(testX,testY)

print("The results of this model are ", 100*score,"% accuracy. ")