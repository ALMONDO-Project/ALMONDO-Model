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

class NoTweetsLeftException(Exception):
    print('>>> The API limit for GET operations was reached. Wait untill the 20th of this month.')
    pass

class NoTweetsToSaveException(Exception):
    print('>>> self.tweets is empty')
    pass

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
        self.end_time = end_time  # Corrected assignment
        self.tweet_fields = tweet_fields
        self.media_fields = media_fields
        self.expansions = expansions
        self.results_per_call = results_per_call
        
        self.max_id = None #potrei dover usare max_id perché non so se sta richiesta me li da dal più recente al più vecchio o viceversa
        self.tweets = {}     
        self.home = os.getcwd()
        self.path = f"{self.home}/data"       
        self.tweets_path = f"{self.path}/out/{username}"
        self.filename = f"{self.tweets_path}/{username}_tweets.json"
        self.search_credentials = load_credentials(filename=f"{self.home}/cred.yaml", yaml_key="search_tweets_cred")
        self.query = f"from:{username} -is:retweet"
    
    def make_dirs(self):
        if not os.path.exists(self.tweets_path):  # Corrected condition
            print(f'>>> creating folder {self.tweets_path}')
            os.makedirs(self.tweets_path)
            print(f'>>> folder {self.tweets_path} created')
        else:
            print(f'>>> {self.tweets_path} is already present')
    
    def set_max_id(self):
        #potrei dover usare max_id perché non so se sta richiesta me li da dal più recente al più vecchio o viceversa
        try:
            self.max_id = get_oldest_tweet_id(self.tweets_path)
            if not self.max_id:
                raise Exception("Empty file exists")
            print(f'>>> oldest tweet downloaded is {self.max_id} at {self.tweets_path}')
        except FileNotFoundError:
            self.max_id = None
            print(f'>>> No previous tweet downloaded, setting max_id to None')
      
    def set_max_tweets(self, BEARER_TOKEN, n = None):
        left = compute_max_tweets(BEARER_TOKEN) - 1000
        if left > 0 and n is None:
            self.max_tweets = left
            print(f'>>> asking to download {self.max_tweets} at max')
        elif left > 0 and n is not None:
            self.max_tweets = n
            print(f'>>> asking to download {n} tweets at max')
        else:
            self.max_tweets = None 
            raise NoTweetsLeftException('No tweets left to download')  # Raised custom exception
        
    def download(self):
        self.search_rule = gen_request_parameters(self.query,
                                        results_per_call=self.results_per_call,
                                        tweet_fields=self.tweet_fields,
                                        media_fields=self.media_fields,
                                        expansions=self.expansions,
                                        max_id = self.max_id, #potrei dover usare max_id perché non so se sta richiesta me li da dal più recente al più vecchio o viceversa
                                        start_time=self.start_time,
                                        end_time=self.end_time)
        
        print(f"Collecting tweets for user {self.filename}")
        
        tweets = collect_results(self.search_rule, max_tweets=self.max_tweets, result_stream_args=self.search_credentials)
        self.tweets = tweets
        
        # Save tweets to file with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename_with_timestamp = f"{self.home}/{self.username}_tweets_{timestamp}.json"
        with open(filename_with_timestamp, 'w') as file:
            json.dump(self.tweets, file)
            
    def save_tweets(self):
        if self.tweets:  # Changed condition to check if tweets is not empty
            with open(self.filename, 'r') as ifile:
                tweet_dict = ifile.read(self.filename)
                progressive_tweet_id = max(tweet_dict.keys(), key=int) + 1

            for tweet in tweet_dict:
                tweet_dict[str(progressive_tweet_id)] = dict(tweet)
                progressive_tweet_id += 1
                
            with open(self.filename, 'w') as ofile:
                json.dump(tweet_dict, ofile)
        else:
            raise NoTweetsToSaveException