import pandas as pd
import numpy as np
import sys  
import re
import urllib.request
import urllib.parse
import json
import csv
import codecs
import os
import socket
from socket import AF_INET, SOCK_DGRAM
from bs4 import BeautifulSoup

class LyricsCrawler:

  # get / manage access token at https://genius.com/api-clients
  client_access_token = 'your_token_here'


  def __init__(self, artists): 
    self.artists = artists
    self.df = pd.DataFrame(columns=['Artist','Title', 'url'])

    for artist in artists:
      artist_df = self.search(artist)
      self.df = pd.merge(self.df, artist_df, how='outer')

    self.get_all_lyrics()

  
  def search(self, search_term):
    """ 
    This function uses the Genius API for a search query on an artist name.
    It outputs a df containing artist name, song title and url to the lyrics.
    (Genius limits the search to about 1000 entries which usually is enough).
    Also does some basic data cleaning: If the primary artist name does 
    not match the search query, than this entry is skipped. 
    Originally from github/jasonqng, adapted for Python 3 and modified.
    """

    page=1
    total_hits = 0
    results = []

    print('Searching genius for '+ search_term + ' ...')
    while True:
        
        querystring = "http://api.genius.com/search?q=" + urllib.parse.quote(search_term) + "&page=" + str(page)
        request = urllib.request.Request(querystring)
        request.add_header("Authorization", "Bearer " + self.client_access_token)   
        request.add_header("User-Agent", "curl/7.9.8 (i686-pc-linux-gnu) libcurl 7.9.8 (OpenSSL 0.9.6b) (ipv6 enabled)") #Must include user agent of some sort, otherwise 403 returned
        
        while True:
            try:
                response = urllib.request.urlopen(request, timeout=4) #timeout set to 4 seconds; automatically retries if times out
                raw = response.read()
            except socket.timeout:
                print("Timeout raised and caught.")
                continue
            break

        json_obj = json.loads(raw)
        body = json_obj["response"]["hits"]

        num_hits = len(body)
        total_hits = total_hits + num_hits
        if num_hits==0:
            if page==1:
                print("No results for: " + search_term)
            break      
    
        for result in body:
            primaryartist_name = result["result"]["primary_artist"]["name"]
            title = result["result"]["title"]
            url = result["result"]["url"]
            row=[primaryartist_name, title, url]
            if primaryartist_name.lower() == search_term.lower():
              results.append(row)
        page+=1

    df=pd.DataFrame(results,columns=['Artist','Title', 'url'])
    print('Search completed, ' + str(total_hits) + ' hits found.')

    return df

  
  def get_lyrics(self, url):
    """
    This function uses the Genius API to extract lyrics from a search result.
    Originally from github/kvsingh, slightly modified.
    """

    request = urllib.request.Request(url)
    request.add_header("Authorization", "Bearer " + self.client_access_token)
    request.add_header("User-Agent",
                       "curl/7.9.8 (i686-pc-linux-gnu) libcurl 7.9.8 (OpenSSL 0.9.6b) (ipv6 enabled)")
    page = urllib.request.urlopen(request)
    soup = BeautifulSoup(page, "lxml")
    lyrics = soup.find("div", class_= "lyrics")

    if lyrics==None:
      return ''
    else:
      return lyrics.text
    
  
  def get_all_lyrics(self):
    """ 
    This function gets the lyrics from all urls in the dataframe.
    """

    print('Getting lyrics for ' + str(len(self.df['url'])) + ' items...')
    print('(This may take a while...)')

    self.df['lyrics'] = ''
    for col, url in self.df['url'].iteritems(): 
      self.df['lyrics'].iloc[col] = self.get_lyrics(url)

    print('Done!')


  def append_artists(self, new_artists):
    """
    This function appends an artist to the current artist list and df.
    (not yet tested)
    """
    new_df = pd.DataFrame(columns=['Artist','Title', 'url'])

    for artist in new_artists:
      artist_df = self.search(artist)
      new_df.append(artist_df)

    for col, url in new_df['url'].iteritems():
      new_df['lyrics'].iloc[col] = self.get_lyrics(url)
      
    self.artists.append(new_artists)
    self.df.append(new_df)
    

  def save(self, outputfilename):
    """
    This function saves the df containing the song lyrics as an *.xlsx-file.
    """
    self.df.to_excel(outputfilename + '.xlsx')
    print(outputfilename + ' saved!')


  def get_df(self):
    """
    This function returns the df containing the song lyrics.
    """
    return self.df