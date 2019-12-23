import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import gensim
from sklearn.feature_extraction.text import CountVectorizer


class TopicModeller:
  """
  This class is doing topic modelling on a given corpus.
  Intended for the JazzLyrics Dataset.
  Attributes:
  config         - dictionary with hyperparameters.
  raw            - original corpus, e.g. a pandas Series.
  stemmed_corpus - corpus after preprocessing (stemming).
  gensim_corpus  - preprocessed corpus converted to gensim corpus.
  vect           - fitted CountVectorizer.
  id_map         - map from ID to word.
  ldamodel       - trained LDA model.
  topic_matrix   - matrix containing the most indicative words for n topics.
  score          - score indicating how distinct the topics are.
  verbose        - 0: print only most important information, 1: print details.
  """

  def __init__(self, config, corpus):
    self.n_topics = config['n_topics']      
    self.n_passes = config['n_passes']      
    self.min_docfreq = config['min_docfreq']  
    self.max_docfreq = config['max_docfreq']
    self.ngrams = config['ngrams']   
    self.n_words = config['n_words'] 
    self.topic_range = config['topic_range']  
    self.ext_stop_words = config['ext_stop_words']
    self.raw = corpus
    self.verbose = 0


  def _preproc(self):
    """
    This function preprocesses the input corpus: 
    * Stop-word Definition
    * Vectorization 
    * Stemming
    * Creation of gensim corpus
    * Creation of ID-map from ID to word.
    """
    # Define stop words
    stop_words = nltk.corpus.stopwords.words('english') 
    stop_words.extend(self.ext_stop_words)

    # Tokenize, stem, then put string back together
    porter = nltk.PorterStemmer()
    self.stemmed_corpus = []
    for idx, string in self.raw.iteritems():
      tokens = nltk.word_tokenize(string)
      stemmed = [porter.stem(t) for t in tokens]
      stemmed_str = ' '.join([str(s) for s in stemmed])
      self.stemmed_corpus.append(stemmed_str)

    # Vectorize corpus with CountVectorizer
    self.vect = CountVectorizer(min_df=self.min_docfreq, 
                           max_df=self.max_docfreq, 
                           stop_words=stop_words, 
                           token_pattern='(?u)\\b\\w\\w\\w\\w+\\b',
                           ngram_range=self.ngrams)

    # Fit and transform
    X = self.vect.fit_transform(self.stemmed_corpus)
    print('Bag-of-Words shape: ', X.shape) # (n_documents, n_vectorized-words)

    # Convert sparse matrix to gensim corpus
    self.gensim_corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

    # Mapping from word IDs to words
    self.id_map = dict((v, k) for k, v in self.vect.vocabulary_.items())


  def _lda(self):
    """
    This function runs LDA for model parameter estimation on the corpus.
    """
    self.ldamodel = gensim.models.ldamodel.LdaModel(self.gensim_corpus, 
                                                    num_topics=self.n_topics, 
                                                    id2word=self.id_map, 
                                                    passes=self.n_passes,
                                                    random_state=42)
    
    self.topic_matrix = self.ldamodel.print_topics(num_topics=self.n_topics, 
                                                   num_words=self.n_words)
  

  def _evaluate(self):
    """
    This function computes the score of the topic distribution.
    As scoring function u_mass coherence is chosen. Compared to the intrinsic
    perplexity score, literature states that coherence yields better results.
    """
    coherence = gensim.models.coherencemodel.CoherenceModel(model=self.ldamodel,
                                                            corpus=self.gensim_corpus,
                                                            dictionary=self.ldamodel.id2word,
                                                            coherence='u_mass')
    self.score = coherence.get_coherence()
    if self.verbose:
      print('LDA achieved a coherence (u_mass) of: ', self.score)


  def run(self, verbose=0):
    """
    This function runs the complete topic modelling pipeline.
    """
    self.verbose = verbose
    self._preproc()
    self._lda()
    self._evaluate()


  def optimize(self):
    """
    This function runs an optimization on the number of topics to evaluate an
    appropriate number of topics via perplexity score.
    """
    scores = []
    n_topics = np.arange(self.topic_range[0], self.topic_range[1]+1)
    print('Running optimization with topic range from {0} to {1}'.format(
      self.topic_range[0],self.topic_range[1]))
    self._preproc()

    # Perform LDA for topic_range
    for n in n_topics:
      self.n_topics = n
      self._lda()
      if self.verbose:
        print('LDA completed for {0} topics.'.format(n))
      self._evaluate()
      scores.append(self.score)
    
    # Visualize results
    print('Optimization completed, plotting results...')
    fig1, ax1 = plt.subplots()
    ax1.plot(n_topics, np.asarray(scores))
    ax1.set_title('Coherence for topic range from {0} to {1}'.format(
      self.topic_range[0], self.topic_range[1]), fontsize= 16)
    ax1.set_xlabel('n_topics')
    ax1.set_ylabel('score')
    ax1.set_xticks(n_topics)
    plt.show()


  def get_lda(self):
    """
    This function returns the most important attributes: 
    ldamodel, topic_matrix and score.
    """
    return self.ldamodel, self.topic_matrix, self.score


  def label(self, input_doc=None):
    """ 
    This function returns the most-relevant topic of a given corpus as a list.
    If called without input argument, it outputs the most-relevant topic of the
    corpus used for training the LDA. 
    Uses pre-trained CountVectorizer with stemmed JazzLyircs Dataset.
    """
    if input_doc == None:
      input_doc = self.stemmed_corpus
    X = self.vect.transform(input_doc)
    new_corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
    topics = self.ldamodel.get_document_topics(new_corpus)
    max_topic = []
    for tpc in list(topics):
      # get most relevant topic (tuple: 0 = topic, 1 = relevance distribution)
      max_topic.append(max(tpc,key=lambda item:item[1])[0]) 
    return max_topic


  def update(self, config):
    """
    This function updates the config, e.g. to change the number of topics after
    optimization.
    """
    self.n_topics = config['n_topics']      
    self.n_passes = config['n_passes']      
    self.min_docfreq = config['min_docfreq']  
    self.max_docfreq = config['max_docfreq']
    self.ngrams = config['ngrams']   
    self.n_words = config['n_words'] 
    self.topic_range = config['topic_range'] 
    self.ext_stop_words = config['ext_stop_words']

  def visualize(self):
    """
    This function uses the pyLDAvis-Visualization to plot the LDA results.
    """
    # TODO
    #pyLDAvis.enable_notebook()
    #vis = pyLDAvis.gensim.prepare(self.lda_model, self.stemmed_corpus)
    return