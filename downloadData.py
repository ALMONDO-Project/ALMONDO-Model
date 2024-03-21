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

class TweetAlreadyDumpedException():
    pass

#class definition

BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAAB6rAEAAAAAUETTBU7ohCCUuzTnRu1VHgYw4Vk%3DN5I6Yq9m16CwhIjwHsFrYx87qsqBeHxwD1lA6bksneT5IlsIvS'

class UserDataDownload():
    def __init__(self,
                 bearer_token = BEARER_TOKEN,
                 username='',
                 expansions='attachments.media_keys,geo.place_id,author_id',
                 tweet_fields='text,id,attachments,created_at,lang,author_id,entities,geo,public_metrics',
                 media_fields='media_key,type,url,variants,preview_image_url',
                 user_fields='id,username,description,public_metrics,verified',
                 start_time =datetime(year=2023, month=1, day=1, hour = 0, minute = 0, second = 0)):
        # Initialize UserDataDownload object with required parameters and default values
        self.bearer_token = bearer_token
        self.username = username.replace('@', '')  # Remove '@' from username if present
        self.tweet_fields = tweet_fields.split(',')  # Split tweet fields into list
        self.media_fields = media_fields.split(',')  # Split media fields into list
        self.expansions = expansions.split(',')  # Split expansions into list
        self.user_fields = user_fields.split(',')  # Split user fields into list
        self.start_time = start_time
        
    def set_client(self, wait_on_rate_limit=True):
        # Set Twitter API client with specified parameters
        self.client = tweepy.Client(self.bearer_token, 
                                    wait_on_rate_limit=wait_on_rate_limit)
        log_message('>>> client initialized')
        
    def set_user_data(self):
        # Get user data using Twitter API client
        self.user = self.client.get_user(username=self.username)
        self.user_id = self.user.data.id

    def make_dirs(self):
        try:
            # Create necessary directories for storing data and logs
            self.datapath = 'data/'
            if not os.path.exists(self.datapath):
                os.makedirs(self.datapath)
            self.outdatapath = 'data/out'
            if not os.path.exists(self.outdatapath):
                os.makedirs(self.outdatapath)
            self.useroutdatapath = f'data/out/{self.username}'
            if not os.path.exists(self.useroutdatapath):
                os.makedirs(self.useroutdatapath)
            self.logpath = 'data/log'
            if not os.path.exists(self.logpath):
                os.makedirs(self.logpath)
            self.userlogpath = f'data/log/{self.username}'
            if not os.path.exists(self.userlogpath):
                os.makedirs(self.userlogpath) 
            log_message('>>> necessary directories created')
        except OSError as e:
            print(e)      
            
    def dump_tweets(self, tweet, tweet_data):
        if not os.path.exists(f'{self.userlogpath}/{tweet.id}.json'):
            with open(f'{self.userlogpath}/{tweet.id}.json', 'w') as file:
                json.dump(tweet_data, file, indent=4, sort_keys=True, default=str)  # Write tweet data to file
        else:
            raise TweetAlreadyDumpedException
        
    def set_limits(self, max_tweets_available, max_tweets_per_session, max_tweets_per_call=100, max_num_calls=None):
        self.max_tweets_available = max_tweets_available
        self.max_tweets_per_session = max_tweets_per_session
        self.max_tweets_per_call = max_tweets_per_call
        self.max_num_calls = max_num_calls
        if not self.max_num_calls:
            self.max_num_calls = math.ceil(self.max_tweets_per_session / self.max_tweets_per_call)
    
    def get_oldest_tweet_date(self):
            filenames = []
            for filename in os.listdir(f'{self.userlogpath}'):
                if filename.endswith('.json') and not filename.startswith('page_meta_'):
                    filenames.append(int(filename.replace('.json', '')))
            filenames.sort(reverse=False)
            if len(filenames) <= 0:
                return datetime(2023, 12, 31, 23, 59, 59)
            oldest_id = filenames[0]
            print(oldest_id)
            oldest_date = ''
            with open(f'{self.userlogpath}/{oldest_id}.json', 'r') as file:
                oldest = json.load(file)
            oldest_date =oldest[str(oldest_id)]['created_at']
            oldest_date = datetime.strptime(oldest_date, '%Y-%m-%dT%H:%M:%S.%fZ')
            oldest_date =oldest_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            return oldest_date
    
    def set_end_time(self):
        self.end_time = self.get_oldest_tweet_date()
                        
    def set_paginator(self):        
        self.set_end_time()   
        self.next_token = self.set_last_pagination_token()
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
                            max_results = self.max_tweets_per_call,
                            pagination_token = self.next_token)
        return self.paginator
    
    def set_last_pagination_token(self):
        # Get a list of all JSON files in the directory
        json_files = [file for file in os.listdir(self.userlogpath) if file.startswith("page_meta_") and file.endswith(".json")]
        # Sort files by modification time (most recent first)
        json_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.userlogpath, x)), reverse=True)
        # Check if there are any JSON files
        if json_files:
            page_file = json_files[0]
            # Open the most recent JSON file
            with open(f'{self.userlogpath}/{page_file}', 'r') as file:
                data = json.load(file)
                # Return the value for the key "next_token"
            page_file = page_file.split('_')
            newest_id = page_file[2]
            oldest_id = page_file[3]
            if os.path.exists(f'{self.userlogpath}/{oldest_id}.json'):
                print('tweets in last saved page have been already dumped, can move to next page')            
                return data.get("next_token", None)
            else:
                print('tweets in this page still need to be retrieved'.capitalize())
                return None
        else:
            # If no JSON files found, return None
            return None
        
    def write_on_file(what, where):
        with open(where, 'w') as file:
            file.write(what)
    
    def get_tweets(self, page):
        if len(page.data) > 0:
            for tweet in page.data: 
                tweet_data = {tweet.data['id']: tweet.data}
                self.dump_tweets(tweet, tweet_data) 
        else:
            raise ValueError("No more tweets to download")
    
    def configureSession(self):
        self.set_client()
        self.set_user_data()
        self.make_dirs()
        paginator = self.set_paginator()
        return paginator
        

    def save_page(self, page):
        if page.data is not None:
            newest_id = page.meta['newest_id']
            oldest_id = page.meta['oldest_id']
            with open(f'{self.userlogpath}/page_meta_{newest_id}_{oldest_id}.json', 'w') as file:
                json.dump(page.meta, file)
            if not page.meta['next_token']:
                self.get_tweets(page)
                raise ValueError("No more pages to download")
        else:
            raise ValueError("Empty page. No more tweets for this user. Moving to next user.")
        
    def download_user_tweets(self):
        paginator = self.configureSession()
        for page in paginator:
            self.save_page(page)
            self.get_tweets(page)    
            time.sleep(2 * 60)