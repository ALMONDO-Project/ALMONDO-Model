#import statements
import sys
import os
import json
import tweepy
import tqdm
import math
import time
import sys
import glob
from datetime import datetime
from utils import *

class LimitsExceededError(Exception): 
    def __init__(self): 
        super().__init__("Max tweet limit exceeded. Set a lower count.")

class TweetAlreadyDumpedException(Exception): 
    def __init__(self): 
        super().__init__("Tweet is already saved in user directory. Moving to next one.")


BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAAB6rAEAAAAAUETTBU7ohCCUuzTnRu1VHgYw4Vk%3DN5I6Yq9m16CwhIjwHsFrYx87qsqBeHxwD1lA6bksneT5IlsIvS'

class UserDataDownload():
    def __init__(self,
                 bearer_token = BEARER_TOKEN,
                 datapath='../data',
                 username='',
                 expansions='attachments.media_keys,geo.place_id,author_id',
                 tweet_fields='text,id,attachments,created_at,lang,author_id,entities,geo,public_metrics',
                 media_fields='media_key,type,url,variants,preview_image_url',
                 user_fields='id,username,description,public_metrics,verified',
                 start_time =None,
                 end_time = None,
                 count = 0):
        # Initialize UserDataDownload object with required parameters and default values
        self.datapath = datapath
        self.bearer_token = bearer_token
        self.username = username.replace('@', '')  # Remove '@' from username if present
        self.tweet_fields = tweet_fields.split(',')  # Split tweet fields into list
        self.media_fields = media_fields.split(',')  # Split media fields into list
        self.expansions = expansions.split(',')  # Split expansions into list
        self.user_fields = user_fields.split(',')  # Split user fields into list
        self.start_time = start_time
        self.end_time = end_time
        self.count = count
        if self.count > compute_max_tweets(BEARER_TOKEN):
            raise LimitsExceededError
        
    def set_client(self, wait_on_rate_limit=True):
        # Set Twitter API client with specified parameters
        self.client = tweepy.Client(self.bearer_token, 
                                    wait_on_rate_limit=wait_on_rate_limit)
        print('>>> client initialized')
        
    def set_user_data(self):
        # Get user data using Twitter API client
        self.user = self.client.get_user(username=self.username)
        print(self.user)
        self.user_id = self.user.data.id
        print(self.user_id)

    def make_dirs(self):
        try:
            # Create necessary directories for storing data and logs
            if not os.path.exists(self.datapath):
                os.makedirs(self.datapath)
            self.logpath = f'{self.datapath}/log'
            if not os.path.exists(self.logpath):
                os.makedirs(self.logpath)
            self.userlogpath = f'{self.logpath}/{self.username}'
            if not os.path.exists(self.userlogpath):
                os.makedirs(self.userlogpath) 
            print('>>> necessary directories created')
        except OSError as e:
            print(e)      
            
    def dump_tweets(self, tweet, tweet_data):
        if not os.path.exists(f'{self.userlogpath}/{tweet.id}.json'):
            with open(f'{self.userlogpath}/{tweet.id}.json', 'w') as file:
                print('>>> tweet dumped')
                json.dump(tweet_data, file, indent=4, sort_keys=True, default=str)  # Write tweet data to file
        elif  os.path.exists(f'{self.userlogpath}/{tweet.id}.json'):
            try:
                with open(f'{self.userlogpath}/{tweet.id}.json', 'r') as file:
                    data = json.load(file)
                    if data:
                        print('Tweet already dumped moving forward')
                        return
            except ValueError:
                print('Existing file is empty. Rewriting the file.')
                with open(f'{self.userlogpath}/{tweet.id}.json', 'w') as file:
                    print('>>> tweet dumped')
                    json.dump(tweet_data, file, indent=4, sort_keys=True, default=str)
                
    def set_limits(self, max_tweets_per_session, max_tweets_per_call=100, max_num_calls=None):
        self.max_tweets_per_session = max_tweets_per_session
        self.max_tweets_per_call = max_tweets_per_call
        self.max_num_calls = max_num_calls
        if not self.max_num_calls:
            self.max_num_calls = math.ceil(self.max_tweets_per_session / self.max_tweets_per_call)
                  
    def set_paginator(self, start_time=None, end_time=None):       
        if end_time is None: 
            self.end_time = datetime(2023,12,31,23,59)
        else:
            self.end_time = end_time   
        if start_time is None: 
            self.start_time = datetime(2023,1,1,0,0)
        else:
            self.start_time = start_time  
        self.paginator = tweepy.Paginator(self.client.get_users_tweets,
                            self.user_id,
                            exclude = ['retweets'],
                            expansions = self.expansions,
                            tweet_fields = self.tweet_fields,
                            media_fields = self.media_fields,
                            user_fields = self.user_fields,
                            start_time = self.start_time,
                            end_time = self.end_time,
                            limit = self.max_num_calls,
                            max_results = self.max_tweets_per_call)
        return self.paginator
    
    def get_tweets(self, page):
        print(page)
        if len(page.data) > 0:
            for tweet in page.data: 
                if self.count > 0:
                    tweet_data = {tweet.data['id']: tweet.data}
                    self.dump_tweets(tweet, tweet_data) 
                    self.count -= 1
                    print(f'{self.count} tweets left to download')
        else:
            raise ValueError("Empty page. No more tweets to download")
    
    def configureSession(self):
        paginator = self.set_paginator()
        return paginator
        
    def download_user_tweets(self):
        paginator = self.configureSession()
        for page in paginator:
            # self.save_page(page)
            self.get_tweets(page)    
            print('>>> going to sleep for 2 minutes')
            for i in tqdm.tqdm(range(60)):
                time.sleep(2)
    
    def get_count(self):
        return self.count
            