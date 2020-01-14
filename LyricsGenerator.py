# LyricsGenerator Class
# Used for jazz style lyrics generation 
# - Preprocesses input corpus (tokenization, encoding, vectorization, ...)
# - creates (or loads pre-trained) keras model
# - trains and saves model
# - generates new song lyrics

from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np
import pandas as pd
import tensorflow as tf


# TensorFlow
%tensorflow_version 2.x
import tensorflow as tf

class LyricsGenerator:
  """
  (work in progress)
  This class is used for training a generative model that outputs Jazz lyrics.
  """

  def __init__(self, config, corpus):

    # Initialize corpus and config
    self.raw                  = corpus
    self.min_freq             = config['min_freq']
    self.mini_batch_len       = config['mini_batch_len']
    self.mini_batch_step      = config['mini_batch_step']
    self.n_units_1            = config['n_units_1']
    self.n_units_2            = config['n_units_2']
    self.dropout_rate         = config['dropout_rate']
    self.l2_reg               = config['l2_reg']
    self.embedding            = config['embedding']
    self.embedding_size       = config['embedding_size']
    self.batch_size           = config['batch_size']
    self.n_epochs             = config['n_epochs']
    self.create_model         = config['create_model']

    # Preprocess corpus (tokenization and word <-> ID maps)
    self._preproc()

    # Make X and Y mini batches
    self._generateMiniBatches()

    # Create Model (and One Hot Encode if no Embedding Layer is used)
    if self.create_model:
      if self.embedding:
        self._createModel()
      else:
        self._oneHotEncode()
        self._createModel()
    return


  def _preproc(self):
    """
    This function preprocesses the input corpus:
      * tokenizes the input corpus
      * creates word-id-maps
      * vectorizes the input corpus
    """

    print('Preprocessing input corpus...')
    # Tokenize corpus using nltk tokenizer
    corpus = self.raw.apply(word_tokenize)

    # Lowercase corpus and get individual words
    for idx, s in corpus.iteritems():
      corpus[idx] = [s.lower() for s in corpus[idx]]

    # Get all words into a list of appropriate shape
    words = []
    for idx, s in corpus.iteritems():
      words.append([s for s in corpus[idx]])
    words = [word for s in words for word in s]
    print('Total Word count:  ', len(words))

    # Remove infrequent words
    counts = Counter(words)
    print('Unique word count: ', len(counts))
    word_dict = {key:val for key, val in counts.items() if val > self.min_freq}

    # Get unique words from corpus and insert <OOV> token
    unique_words = sorted(word_dict.keys())
    unique_words.insert(0,'<OOV>')

    # Create word <-> id maps
    self.word_id = dict((w, i) for i, w in enumerate(unique_words))
    self.id_word = dict((i, w) for i, w in enumerate(unique_words))
    self.idmap_len = len(self.word_id)
    print('Vocabulary size:   ', self.idmap_len)

    # Vectorize corpus
    self.tok_corpus = []
    for idx, s in corpus.iteritems():
      document = [word for word in corpus[idx]]
      tok_doc = []
      for word in document:
        if word in self.word_id.keys(): 
          tok_doc.append(self.word_id[word])
        else: 
          tok_doc.append(self.word_id['<OOV>'])
      self.tok_corpus.append(tok_doc)
    return 


  def _generateMiniBatches(self):
    """
    This function generates semi-redundant mini batches that can be used 
    for training.
      * The shape is (m, T_mini, n) where T_mini is the window size.
      * Mini-batches are dilated with given step size.
    """

    # Init array
    self.X_mini = []
    self.Y_mini_s = [] # Use 10 songs 
    print('Generating mini-batches...')

    # iterate over tokenized corpus
    for song in self.tok_corpus:
      # Zero-pad beginning of each song with "batch_size-1" <OOV> tokens
      song = [self.word_id['<OOV>']]*(self.mini_batch_len-1) + song

      # iterate over one song from tokenized corpus to create mini-batches
      for i in range(0, len(song) - (self.mini_batch_len+1), self.mini_batch_step):
        self.X_mini.append(song[i: i + self.mini_batch_len])
        self.Y_mini_s.append(song[i+self.mini_batch_len+1])

    # Convert to array in correct shape
    self.X_mini = np.asarray(self.X_mini)
    self.X_mini = np.expand_dims(self.X_mini, axis=2)
    self.Y_mini = np.asarray([np.append(x[1:], self.word_id['<OOV>']) for x in self.X_mini])
    self.Y_mini = np.expand_dims(self.Y_mini, axis=2)
    self.Y_mini_s = np.asarray(self.Y_mini_s)
    self.Y_mini_s = np.expand_dims(self.Y_mini_s, axis=1)

    # Print output shapes
    print('Shape of X_mini:   ', self.X_mini.shape)
    print('Shape of Y_mini:   ', self.Y_mini.shape)
    print('Shape of Y_mini_s: ', self.Y_mini_s.shape)
    return 


  def _oneHotEncode(self):
    """
    This function takes the mini batch arrays and one hot encodes the tokens.
    The array will be in shape (m, T, n), where:
      * m: number of samples
      * T: sample length, i.e. length of mini batch windows
      * n: number of features, i.e. length of word-id-mapping
    """

    print('Encoding mini-batches...')

    # Initialize OHE arrays
    n_samples = self.X_mini.shape[0]
    max_len = self.mini_batch_len
    vocab_len = self.idmap_len
    self.X_onehot = np.zeros((n_samples, max_len, vocab_len), dtype=np.bool)
    self.Y_onehot = np.zeros((n_samples, max_len, vocab_len), dtype=np.bool)
  
    # one-hot-encode X
    for i, song in enumerate(self.X_mini):
      for t, word in enumerate(song):
        self.X_onehot[i, t, word] = 1

    # one-hot-encode Y    
    for i, song in enumerate(self.Y_mini):
      for t, word in enumerate(song):
        self.Y_onehot[i, t, word] = 1

    print('Shape of X_onehot:  ', self.X_onehot.shape)
    print('Shape of Y_onehot:  ', self.Y_onehot.shape)    
    return  


  def _createModel(self):
    """
    This function creates a keras sequential model for lyrics generation.
    Depending on the config, it adds an Embedding as first layer.
    Initially, I'll use a fixed architecture with the number of units and 
    regularization rates as hyperparameters. 
    """
    
    print('Creating keras model...')

    self.model = tf.keras.models.Sequential()
    
    # Use Embedding Layer before Bidirectional if selected in config
    if self.embedding:
      self.model.add(tf.keras.layers.Embedding(self.idmap_len, 
                                               self.embedding_size,
                                               input_length=self.mini_batch_len))
      self.model.add(tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(self.n_units_1, return_sequences=True))) 
       
    # Else use Bidirectional as first Layer    
    else:
      self.model.add(tf.keras.layers.Bidirectional(
                   tf.keras.layers.LSTM(self.n_units_1, return_sequences=True),
                                        input_shape=(self.mini_batch_len, 
                                                     self.idmap_len)))
      
    # Following Layers (independenant of Embedding)
    self.model.add(tf.keras.layers.Dropout(self.dropout_rate))
    self.model.add(tf.keras.layers.LSTM(self.n_units_2, return_sequences=False))
    self.model.add(tf.keras.layers.Dense(self.idmap_len/64, 
                                        activation='relu', 
                                        kernel_regularizer=
                                         tf.keras.regularizers.l2(self.l2_reg)))
    self.model.add(tf.keras.layers.Dense(self.idmap_len, 
                                        activation='softmax'))

    self.model.compile(loss='sparse_categorical_crossentropy', 
                       optimizer='adam', 
                       metrics=['accuracy'])
    
    self.model.summary()
    return


  def train(self):
    """
    This function is used for training the model.
    Currently works only for Embedding version.
    """

    print('Training model...')
    # TODO: Add on_epoch_end Callback to see training progress
    history = self.model.fit(self.X_mini[:,:,0], 
                             self.Y_mini_s[:,0], 
                             batch_size=self.batch_size, 
                             epochs=self.n_epochs, 
                             verbose=2)


  def sample(self, seed_text, text_len):
    """
    This function generates a jazz song with text_len words and 
    seed_text as first line.
    seed_text must not contain <OOV> tokens.
    """

    print('Generating lyrics with', text_len, 'additional words...')
    # tokenize seed_text
    seed_text = word_tokenize(seed_text)
    seed_toks = [self.word_id[word.lower()] for word in seed_text]

    # fill seed text length to 10 tokens
    if len(seed_toks) < 10:
      seed_toks = [0] * abs(len(seed_toks)-10) + seed_toks

    counter = 0
    oov_counter = 0

    # generate new text
    while counter < text_len:
      # get propability distribution for next word, use only last x words
      # shape of y_pred is (x, 1, vocab_size)
      next_seed = np.asarray(seed_toks[-10:]).reshape(1,10)
      y_pred = self.model.predict(next_seed) 

      # Only for debugging the model and sampling process!
      if counter < 4: 
        print('----', counter, '----')
        for pred in y_pred:  
          print('Max prob val: {0:.3f} and wrd: {1}'.format(np.amax(pred), 
                                                    self.id_word[np.argmax(pred)]))

      # random choice from distribution
      idx = np.random.choice(np.arange(self.idmap_len), p = y_pred.ravel())

      # skip <OOV> words, abort sampling after x consecutive <OOV> tokens
      if idx == self.word_id['<OOV>']:
        oov_counter = oov_counter+1
        if oov_counter > 100:
          print('<OOV> counter overflow! Aborting...')
          break   
        continue
      # reset <OOV> counter after sampling non <OOV> token
      else:
        oov_counter = 0 

      # append token and text
      seed_toks.append(idx)
      seed_text.append(self.id_word[idx])

      # increment counter
      counter = counter+1

    song_text = ' '.join(seed_text)
    print('--------')
    print('Generated song lyrics:')
    print(song_text)
    return 


  def save(self, filepath):
    """
    This function saves a keras model (architecture + weights) as h5-file.
    """
    print('Saving model...')
    self.model.save(filepath)
    print('done!')
    return


  def load(self, filepath):
    """
    This function loads a (pretrained) keras model (architecture + weights).
    """
    print('Loading model...')
    self.model = tf.keras.models.load_model(filepath)
    self.model.summary()
    return