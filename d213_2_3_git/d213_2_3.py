#!/usr/bin/env python
# coding: utf-8

# # D213 Task 2 Rev 3 - Mattinson

# ## imports

# In[1]:


# import required libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from numpy import array
from keras import models
from keras import layers
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

print('tensorflow ver: {}'.format(tf.__version__))
print('nltk ver: {}'.format(nltk.__version__))
print('wordcloud ver: {}'.format(wordcloud.__version__))
print('numpy ver: {}'.format(np.__version__))
print('pandas ver: {}'.format(pd.__version__))
#print('matplotlib ver: {}'.format(plt.__version__))


# ## get data

# In[2]:


# read csv data
amazon = 'data/amazon_cells_labelled.txt'
imdb =  'data/imdb_labelled.txt'
yelp =  'data/yelp_labelled.txt'
colnames=['text', 'label'] 
amazon_df = pd.read_csv(amazon, sep='\t', names=colnames, header=None)
imdb_df = pd.read_csv(imdb, sep='\t', names=colnames, header=None)
yelp_df = pd.read_csv(yelp, sep='\t', names=colnames, header=None)
df = pd.concat([amazon_df, imdb_df, yelp_df])
df = df.reset_index(drop=True)

print('{}\n{}'.format(df.info(), df.shape))
df.sample(5, random_state=0) # 5 random (0) rows of data


# In[3]:


# plot scores as bar plot
print(df['label'].value_counts()) # output to notebook
pd.value_counts(df['label']).plot.bar() # create plot


# In[4]:


# look at 'good' in a negative context
df[(df['text'].str.contains('good') >= 1) & (df['label'] == 0 )]


# In[5]:


# look at 'great' in a negative context
df[(df['text'].str.contains('great') >= 1) & (df['label'] == 0 )]


# In[6]:


# look at 'bad' in a positive context
df[(df['text'].str.contains('bad') >= 1) & (df['label'] == 1 )]


# ## explore data

# In[7]:


# descriptive stattics
print(type(df['label']))
print(df['label'].info())
df.describe()


# ## clean data

# In[8]:


# retype label data


# In[9]:


# remove punctuation
def remove_punctuation(text: str) -> str:
    '''remove punctuation from text'''
    final = "".join(u for u in text if u not in (
        "?", ".", ";", ":", "!", '"', ','))
    return final # updated string
print('before: {}'.format(df['text'].loc[0]))
df['text'] = df['text'].apply(remove_punctuation)
print('\nafter: {}'.format(df['text'].loc[0]))


# In[10]:


# lower case
print('before: {}'.format(df['text'].loc[0]))
df['text'] = df['text'].astype(str).str.lower()
print('\nafter: {}'.format(df['text'].loc[0]))


# In[11]:


# first tokenization
from nltk.tokenize import RegexpTokenizer
regexp = RegexpTokenizer('\w+')
print('before: {}'.format(df['text'].loc[0]))
df['text_token']=df['text'].apply(regexp.tokenize)
print('\nafter: {}'.format(df['text_token'].loc[0]))

# what is type of the new field
print('\ntext_token type: {}'.format(type(df['text_token'])))


# In[12]:


# remove stopwords
stopwords = nltk.corpus.stopwords.words("english")
print(stopwords[0:20]) # just first 20 stopwords...
#my_stopwords = ['https', 'good', 'great', 'bad']
my_stopwords = ['https']
stopwords.extend(my_stopwords)
print('\nbefore: {}'.format(df['text'].loc[0]))
df['text_token'] = df['text_token'].apply(
    lambda x: [item for item in x if item not in stopwords])
print('\nafter: {}'.format(df['text_token'].loc[0]))


# In[13]:


# remove infrequent words
df['text_string'] = df['text_token'].apply(
    lambda x: ' '.join([item for item in x if len(item)>2]))
all_words = ' '.join([word for word in df['text_string']])
tokenized_words = nltk.tokenize.word_tokenize(all_words)
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_words)
print(fdist)
cutoff = 1 # drop words occurring less than certain amount
print('\nbefore: {}'.format(df['text'].loc[0]))
df['text_string_fdist'] = df['text_token'].apply(
    lambda x: ' '.join([item for item in x if fdist[item] >= cutoff ]))
print('\nafter (text_string): {}'.format(df['text_string'].loc[0]))
print('\nafter (text_string_fdist): {}'.format(df['text_string_fdist'].loc[0]))


# In[14]:


# lemmatize 
wordnet_lem = WordNetLemmatizer()
print('\nbefore: {}'.format(df['text'].loc[0]))
df['text_string_lem'] = df['text_string_fdist'].apply(wordnet_lem.lemmatize)
print('\nafter (text_string_lem): {}'.format(df['text_string_lem'].loc[0]))


# In[15]:


# Defining our word cloud drawing function
# adapted from Assaker (2022)
# https://github.com/JosephAssaker/Twitter-Sentiment-Analysis-Classical-Approach-VS-Deep-Learning/blob/master/Twitter%20Sentiment%20Analysis%20-%20Classical%20Approach%20VS%20Deep%20Learning.ipynb
def plot_wordcloud(title: str, data, color = 'black'):
    print(title) # output to notebook
    wordcloud = WordCloud(stopwords = STOPWORDS,
                          background_color = color,
                          width = 2500,
                          height = 2000
                         ).generate(' '.join(data))
    plt.figure(1, figsize = (13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show() # create output plot


# In[16]:


# finding most common words
n_common = 20
all_words_lem = ' '.join([word for word in df['text_string_lem']])
words = nltk.word_tokenize(all_words_lem)
fd = FreqDist(words)
top_x_words = fd.most_common(n_common)
fdist = pd.Series(dict(top_x_words)) # data converted to series
import seaborn as sns
sns.set_theme(style="ticks")
sns.barplot(y=fdist.index, x=fdist.values, color='blue');
print(fd.most_common(n_common))


# https://www.kirenz.com/post/2021-12-11-text-mining-and-sentiment-analysis-with-nltk-and-pandas-in-python/text-mining-and-sentiment-analysis-with-nltk-and-pandas-in-python/

# In[17]:


# wordcloud
wordcloud = WordCloud(width=600, 
                     height=400, 
                     random_state=2, 
                     max_font_size=100).generate(all_words_lem)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');


# ## export clean data

# In[18]:


# review what the data looks like after cleaning
print('{}\n{}'.format(df.info(), df.shape))
df.sample(3, random_state=0) # 5 random (0) rows of data


# In[19]:


# export clean data
f = 'tables\clean.csv'
df.to_csv(f, index=True, header=True)


# ## train test split

# https://www.kaggle.com/code/arunkumarramanan/awesome-ml-and-text-classification-movie-reviews

# ### seed=

# ### test_split=

# In[20]:


# train test split
X = df['text_string_lem']
y = df['label']
seed = 42 # try different seeds
test_split = 0.05 # 0.2 best so far
X_train, X_test, y_train, y_test = tts(X, y, 
        test_size=test_split, random_state=seed)
print(X_train[0:3]) # df['text_string_lem']
print('X_train shape-type: {}-{}'.format(X_train.shape, type(X_train)))
print('X_test shape: {}'.format(X_test.shape))
print('y_train shape-type: {}-{}'.format(y_train.shape, type(y_train)))
print('y_test shape: {}'.format(y_test.shape))


# ## model #1 - keras(Sequential)

# ### n_token_words =

# In[21]:


# second tokenizer words -> numbers
n_token_words = 4425 # best so far = 5000
tokenizer = Tokenizer(num_words=n_token_words)
#print('\ntype: {}\nbefore:\n{}'.format(type(X_test), X_test[0]))
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train) # ndarry/df -> list
X_test = tokenizer.texts_to_sequences(X_test)
#print('\ntype: {}\nafter:\n{}'.format(type(X_test), X_test[0])) # now a list


# In[22]:


print(type(X_test))


# In[23]:


#X_train[0:3] # tokenized


# ### vocab_size =

# ### maxlen = 

# In[24]:


# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1
maxlen = 64
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
np.set_printoptions(threshold=np.inf)
print('vocab_size: {}'.format(vocab_size))
print('maxlen: {}'.format(maxlen))
X_test[0] # now a padded list


# In[25]:


# reset options
#pd.reset_option('all') 


# In[26]:


#X_train[0:3] # padded


# ### dropout = 

# ### output_dim =

# In[27]:


# define model
dropout = 0.4 # use dropout = 0 to specify not dropout layer
output_dim = 2000 # vocab_size # 1-1 mapping to vocab word
model = models.Sequential()
model.add(layers.Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=maxlen))
if(dropout > 0):
    model.add(layers.Dropout(dropout))
model.add(layers.Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())


# In[28]:


# compile model
model.compile(
            optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['acc'])


# In[29]:


# save model in SavedModel format
# prior to saving the model, you need to compile the model
from datetime import datetime
now = datetime.now() # current date and time
date_time_stamp = now.strftime("_%y%m%d_%H%M")
model.save('models/final' + date_time_stamp)


# ### val_split = 

# In[30]:


val_split = .2 # .3 or .4 working best so far
len(X_train)
val_split = int(val_split * len(X_train))
x_val = X_train[:val_split]
partial_x_train = X_train[val_split:]
y_val = y_train[:val_split]
partial_y_train = y_train[val_split:]


# ### batch_size =

# ### n_epochs = 

# In[31]:


batch_size = 32 # 256 best so far
n_epochs = 300 # 100-200 best so far
history = model.fit(partial_x_train,
                    partial_y_train,
                    batch_size=batch_size, 
                    epochs=n_epochs,
                    verbose=0, 
                    validation_data=(x_val, y_val))


# "Usually training should be better than validation..."

# validation loss goes down but then increases - overfit

# ## custom_loss_acc_plot

# In[32]:


import matplotlib.pyplot as plt
import matplotlib.axes as ax
# adapted from Assaker (2022)
def custom_loss_acc_plot(
    ax: ax, 
    hist: dict, 
    title: str,
    n_epochs: int,
    batch_size: int,
    vocab_size: int,
    output_dim: int,
    test_split: int,
    val_split: int,
    maxlen: int,
    seed: int,
    summary: str,
    top: int,
    score: np.ndarray,
    n_token_words: int,
    dropout: float
) -> ax:
    """
    custom subplot returns
    """
    # plot loss on axis=0
    y1 = hist['loss'] 
    y2 = hist['val_loss']
    x = range(1, len(y1) + 1) # x-axis = Epochs 
    ax[0].plot(x, y1, 'b+', label='Training loss')
    ax[0].plot(x, y2, 'b', label='Validation loss')
    ax[0].set_title('Loss')
    ax[0].text(.05 * n_epochs, top - .5, 'n_epochs: ' + str(n_epochs), fontsize=10) 
    ax[0].text(.05 * n_epochs, top - .8, 'batch_size: ' + str(batch_size), fontsize=10) 
    ax[0].text(.05 * n_epochs, top - 1.1, 'vocab_size: ' + str(vocab_size), fontsize=10)
    ax[0].text(.05 * n_epochs, top - 1.4, 'test_split: ' + str(test_split), fontsize=10) 
    ax[0].text(.05 * n_epochs, top - 1.7, 'val_split: ' + str(val_split), fontsize=10) 
    ax[0].text(.05 * n_epochs, top - 2.0, 'maxlen: ' + str(maxlen), fontsize=10) 
    ax[0].text(.05 * n_epochs, top - 2.3, 'seed: ' + str(seed), fontsize=10)
    ax[0].text(.05 * n_epochs, top - 2.6, 'test scores: ' + str(score), fontsize=10)
    ax[0].text(.05 * n_epochs, top - 2.9, 'output_dim: ' + str(output_dim), fontsize=10)
    ax[0].text(.05 * n_epochs, top - 3.2, 'n_token_words: ' + str(n_token_words), fontsize=10)
    if(dropout > 0):
        ax[0].text(.05 * n_epochs, top - 3.5, 'dropout: ' + str(dropout), fontsize=10)
    ax[0].grid(True)
    ax[0].axis('on') 
    ax[0].set_ylim(0,5)
    #ax[0].set_ylim(0,1)
    #ax[0].yaxis.set_major_locator((integer=True))
    ax[0].legend()   

    # plot acc on axis=1
    y1 = hist['acc'] 
    y2 = hist['val_acc']
    x = range(1, len(y1) + 1) # x-axis = Epochs 
    ax[1].plot(x, y1, 'b+', label='Training acc')
    ax[1].plot(x, y2, 'b', label='Validation acc')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].grid(True)
    ax[1].axis('on')
    ax[1].legend()
    
    # plot model summary on axis=2  
    ax[2].text(0, -.2, summary, fontsize=10) 
    ax[2].grid(False)
    ax[2].axis('off')    
    return (ax)

title = 'Loss-Accuracy'
fig, ax = plt.subplots(3, sharex=False, figsize=(7,10))
stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)
score = model.evaluate(X_test, y_test, verbose=0)
top = 5
custom_loss_acc_plot(
    ax, 
    history.history, 
    title,
    n_epochs,
    batch_size,
    vocab_size,
    output_dim,
    test_split,
    val_split,
    maxlen,
    seed,
    short_model_summary,
    top,
    score,
    n_token_words,
    dropout
)

from datetime import datetime
now = datetime.now() # current date and time
title += now.strftime("_%y%m%d_%H%M")
ax[0].text(.05 * n_epochs, top, title, fontsize=12) 
fig.savefig('figures\\' + title, dpi=150)
plt.close()


# ## end of notebook

# In[33]:


# beeps to indicate end of notebook
import winsound
n_beeps = int((score[1]*10-5))
for i in range(5):
    winsound.Beep(700, 100)
for i in range(n_beeps):
    winsound.Beep(500, 200)


# In[ ]:




