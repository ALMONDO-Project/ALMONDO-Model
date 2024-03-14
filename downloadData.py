from searchtweets import collect_results, load_credentials, gen_request_parameters
import tweepy
from utils import *
from datetime import datetime, date, timedelta
import pandas as pd
import json 
import requests
import os
from os.path import join as join, dirname
import smtplib, ssl
from dotenv import load_dotenv

PROJECT_HOME = "C:/Users/valen/GitHub/almondo-tweets-retrieval" 

class UserDataDownloader():
    def __init__(self, 
                 username, 
                 start_time=(datetime(year=2023, month=1, day=1)).strftime("%Y-%m-%d"), 
                 end_time=(datetime(year=2023, month=12, day=31)).strftime("%Y-%m-%d"), 
                 tweet_fields='text,id,attachments,created_at,lang,author_id,entities,geo', 
                 expansions='attachments.media_keys,geo.place_id', 
                 media_fields='media_key,type,url,variants,preview_image_url', 
                 results_per_call=100):

        self.username = username
        self.start_time = start_time
        self.start_time = end_time
        self.tweet_fields = tweet_fields
        self.media_fields = media_fields
        self.expansions = expansions
        self.results_per_call = results_per_call
        self.since_id = None
        self.tweets = {}      
        self.path = f"{PROJECT_HOME}/data"       
        self.tweets_path = f"{self.path}/out/{username}"
        self.filename = f"{self.tweets_path}/{username}_tweets.json"
        self.search_credentials = load_credentials(filename=f"{PROJECT_HOME}/cred.yaml", yaml_key="search_tweets_cred")
        self.query = f"from:{username} -is:retweet"
    
    def make_dirs(self):
        if not os.makedirs(self.tweets_path):
            print(f'>>> creating folder {self.tweets_path}')
            os.makedirs(self.tweets_path)
            print(f'>>> folder {self.tweets_path} created')
        else:
            print(f'>>> {self.tweets_path} is already present')
    
    def set_since_id(self): #controllare che questo sia giusto non mi ricordo se è since_id o l'altro
        try:
            self.since_id = get_last_tweet_id(self.tweets_path)
            print(f'>>> oldest tweet downloaded is {self.since_id} at {self.tweets_path}') #come faccio a farmi dire se non ho altri tweet da scaricare?
        except: 
            self.since_id = None
      
    def set_max_tweets(self, BEARER_TOKEN):
        left = compute_max_tweets(BEARER_TOKEN) - 1000
        if left > 0:
            self.max_tweets = left
        else:
            self.max_tweets = None #se questo è il caso si deve interrompere tutto e me lo deve dire che non ho più tweet da scaricare
            raise Exception('No tweets left to download')
        
    def download(self):
        self.search_rule = gen_request_parameters(self.query,
                                    results_per_call=self.results_per_call,
                                    tweet_fields=self.tweet_fields,
                                    media_fields=self.media_fields,
                                    expansions=self.expansions,
                                    since_id = self.since_id,
                                    start_time=self.start_time,
                                    end_time=self.end_time)
        
        print(f"Collecting tweets for user {self.filename}")
        tweets = collect_results(self.search_rule, max_tweets=self.max_tweets, result_stream_args=self.search_credentials)
        self.tweets = tweets
            
    def save_tweets(self):
        if self.tweets is not None:
            with open(self.filename, 'a+') as ofile:
                ofile.write(self.tweets)           

usernames = read_users(f'PROJECT_HOME/data/in/')

for username in usernames:   
    username = username.replace('@', '')     
    downloader = UserDataDownloader(username)
    downloader.make_dirs()
    downloader.set_since_id()
    downloader.set_max_tweets()
    downloader.download()
    downloader.save_tweets()
    print(f'>>> process ended for user {username}')
print('>>> process ended')