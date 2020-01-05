# DeepJazz-Lyrics
Analysis of the JazzLyrics Dataset.

The five (or five of the) ["best Jazz singers of all time"](https://www.udiscovermusic.com/stories/50-best-jazz-singers/) are Ella Fitzgerald, Frank Sinatra, Nat King Cole, Billie Holiday and Sarah Vaughan. Together, these singers cover most of the songs that today are considered to be [Jazz Standards](https://en.wikipedia.org/wiki/Jazz_standard). 

Jazz Standards are performed and re-interpreted by both amateur and professional musicians around the world. Take, for example, the song "autumn leaves". There are millions of versions of this song, including a rather [traditional version from Frank Sinatra](https://youtu.be/AO-H9Ni5NiQ?t=40), [Sarah Vaughn scatting](https://youtu.be/5cZG2WnXPgk?t=40) on the chords, an instrumental interpretation by the [famous guitar player Joe Pass](https://youtu.be/795sG19cPmU) and again a more traditional version by the contemporary artist [Eric Clapton](https://youtu.be/UQlFOX0YKlQ). 

Looking at all of the artists songs can be considered representative for the 20th century jazz. The goal of this notebook is to analyse the lyrics of these jazz singers. 

First, the dataset is created by crawling through Genius using the Genius API. Then, we will look at some topic modelling and clustering approaches of the songs. Jazz lyrics are short and very poetic, so finding distinct topics will be difficult. Finally, the goal is to create a generative deep neural netwok that outputs a jazz songtext - and record it! :) Future plans include creating a model of the chords (leadsheet) as well.   

## Create Dataset via Genius API

The dataset is created by crawling through Genius. There is a Genius API that can be used to query the site. I've written a class with some more functionality. Inputting a list of artists, LyricsCrawler searches Genius, gets the link to the lyrics and uses BeautifulSoup to parse the web page in order to output the song text. The results can be saved to Excel and it can be returned as a pandas Dataframe. 

```
token = 'your_token_here' # Get your token at https://genius.com/api-clients
artists = ['Frank Sinatra', 'Ella Fitzgerald', 'Nat King Cole', 'Billie Holiday', 'Sarah Vaughan']

lyrics = LyricsCrawler(token, artists)
lyrics.save('JazzLyrics')

df = lyrics.get_df()
```
The resulting dataframe already is quite clean, so very few data cleaning is necessary. Dropping duplicates is recommended (it is very common in Jazz that songs are performed and recorded by multiple artists), and also some basic string cleaning, e.g. to remove metainfo like (Verse 1).

## Topic Modelling: Is Jazz all about Love?

One goal of the Analysis of the JazzLyrics Dataset is topic modelling. To do so, I'll use Latent Dirichlet Allocation (LDA), an algorithm that takes in a given number of topics and a corpus and outputs relevant words (or n-grams) for each topic. Then it can be used to output a topic distribution for each song. I'm taking the most-relevant topic as label for that song. 

I've written a class TopicModeller, which uses LDA but also has some built-in preprocessing functionality like tokenizing and stemming the documents. Also, it can be used to "optimize" the number of topics using the coherence score. Ideally, the coherence score converges at a given number of topics. However, application-specific deviations from this "ideal number of topics" are quite common, as in my case I don't want more than 5 different topics to keep it clear.

```
# Hyperparameters for topic modelling
lda_config = {
'n_topics': 4,         # how many topics
'n_passes': 25,        # how many passes for modelling
'min_docfreq': 25,     # minimum document frequency to include in vectorizer
'max_docfreq': 0.7,    # maximum document frequency to include in vectorizer
'ngrams': (1,2),       # n-gram range, e.g. (1,1) means only single words
'n_words': 5,          # how many topic-related words are displayed
'topic_range': (2,20), # topic range for optimizing number of topics
'ext_stop_words': ['just', 'like', 'know'], # additional custom stop words
}

topicmodel = TopicModeller(lda_config, df['lyrics'])
topicmodel.run(verbose=1)
topicmodel.optimize()
```

Clearly, there is one dominating topic in Jazz songs which is love. It can be difficult to use LDA to extract distinct and interpretable topic-related words, so first I'll manually label love songs simply by looking if the song contains the word "love", which is about 60% of all songs. 

![Love songs](/pics/love_labels.png)

Plotting a confusion matrix of the manually labelled love songs and an LDA running on the whole corpus, the true positive rate of the manual labels vs. the LDA labels is quite high and the F1-score is 0.74. 

![Confusion Matrix](/pics/confusion_matrix.png)

So I have decided to keep the manually labelled love songs and run an LDA on the other songs. The resulting categories are quite good:

```
Bag-of-Words shape:  (994, 679)
[(0,
  '0.022*"babi" + 0.022*"good" + 0.017*"kiss" + 0.014*"right" + 0.014*"tell"'),
 (1,
  '0.022*"time" + 0.019*"come" + 0.018*"heart" + 0.017*"never" + 0.015*"long"'),
 (2,
  '0.037*"dream" + 0.018*"come" + 0.016*"star" + 0.015*"night" + 0.015*"could"'),
 (3,
  '0.029*"blue" + 0.024*"make" + 0.021*"song" + 0.020*"come" + 0.019*"swing"')]
```

![All Songs](/pics/all_labels.png)

Here are some examples where the categories are matching very well. Not all songs are categorized this good, but given the small and poetic dataset, I'm quite content with the results so far.

Love | Baby/Girl | Time | Dreams | Sing & Swing
------------ | ------------- | ------------- | ------------- | ------------- 
Fly Me to the Moon| My Baby Just Cares For Me | The Summer Knows | Dreamy | Sing, Song, Swing
Tenderly | What a Funny Girl (You Used to Be) | Sunday, Monday or Always | Perdido | Get Your Kicks On Route 66
So In Love | I’m Sorry I Made You Cry 	 | Spring Will Be A Little Late This Year 	 | The Nearness Of You | Brahm’s Lullaby

## Generative Model

**In Progress!** 

The goal is to implement a generative neural network that outputs a jazz song and then record it! :)

Currently, I'm working on a class for lyrics generation. Preprocessing, architectural choices of the model (such as whether to use a word or character level basis) and more can be found in the notebook. I'll update this readme once the model outputs first results.
