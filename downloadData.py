#import statements
import os
import json
import tweepy
from utils import *

#class definition

class UserDataDownload():
    def __init__(self,
                 username='',
                 tweet_fields='text,id,attachments,created_at,lang,author_id,entities,geo,public_metrics',
                 expansions='attachments.media_keys,geo.place_id',
                 media_fields='media_key,type,url,variants,preview_image_url',
                 user_fields='id,username,description,public_metrics,verified',
                 until_id = None,
                 max_tweets = 5,
                 limit=100):
        # Initialize UserDataDownload object with required parameters and default values
        self.username = username.replace('@', '')  # Remove '@' from username if present
        self.tweet_fields = tweet_fields.split(',')  # Split tweet fields into list
        self.media_fields = media_fields.split(',')  # Split media fields into list
        self.expansions = expansions.split(',')  # Split expansions into list
        self.user_fields = user_fields.split(',')  # Split user fields into list
        self.until_id = until_id  # Set until_id for pagination
        self.max_tweets = max_tweets  # Maximum number of tweets to retrieve
        self.limit = limit  # API request limit
        self.max_id = None  # Initialize max_id for pagination
        self.tweets = {}  # Dictionary to store tweets
        
    def set_client(self, bearer_token, wait_on_rate_limit=True):
        # Set Twitter API client with specified parameters
        self.client = tweepy.Client(bearer_token, 
                                    expansions = self.expansions,
                                    media_fields = self.media_fields,
                                    tweet_fields = self.tweet_fields,
                                    user_fields = self.user_fields,
                                    wait_on_rate_limit=wait_on_rate_limit)
        
    def set_user_data(self):
        # Get user data using Twitter API client
        self.user = self.client.get_user(username=self.username)
        self.user_id = self.user.data.id
        
    def set_until_id(self):
        # Set until_id for pagination based on existing data if available
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as file:
                d = json.load(file)
            until_id = d[max(int(k) for k in d.keys())]['id'] if d else None
            self.until_id = until_id
        else:
            self.until_id = None
        
    def set_paginator(self):
        # Initialize paginator for fetching user tweets
        self.paginator = tweepy.Paginator(self.client.get_users_tweets,
                                           self.user_id,
                                           until_id = self.until_id,
                                           max_results=self.max_tweets)
    
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
        except OSError as e:
            print(e)
            
    def set_output_filename(self):
        # Set output filename for storing tweets
        self.make_dirs()
        self.filename = os.path.join(self.useroutdatapath, f'{self.username}_tweets.json')
            
    def set_max_tweets(self, bearer_token, n=None):
        # Set the maximum number of tweets to download
        left = compute_max_tweets(bearer_token)  # Compute remaining tweets allowed
        print(f'>>> there are {left} tweets left to download')
        self.max_tweets = min(left, n) if n is not None else left  # Set max_tweets to minimum of remaining tweets and n
        print(f'Asking to download {self.max_tweets} tweets at max')
        if left <= 0:
            print('no tweets left')
            return None

    def download(self):
        self.tweets = {}  # Reset tweets dictionary
        exc = None  # Initialize exception variable
        try: 
            for i, tweet in enumerate(self.paginator.flatten()):  # Iterate through paginated tweets
                self.tweets[i] = dict(tweet)  # Store tweet data in dictionary
                with open(f'{self.filename}', 'a+') as file:
                    json.dump(self.tweets[i], file)  # Write tweet data to file
            print(f'{i} tweets downloaded')
        except Exception as e:
            exc = str(e)  # Store the exception message
        
        # Update the until_id only if tweets were retrieved successfully
        if self.tweets:
            self.until_id = self.tweets[max(self.tweets.keys())]['id']
        
        # Write download log
        d = {'user': self.username, 'user_id': self.user_id, 'last_tweet_id': self.until_id, 'exception': exc}
        with open(f'{self.userlogpath}/downloadlog.json', 'a') as file:
            json.dump(d, file)
        
        return self.tweets
        
    def save(self):
        # Save tweets data to file
        with open(f'{self.filename}', 'a') as file:
            json.dump(self.tweets, file)


''' 
Example:
BEARER_TOKEN = 'some string indicating the token'
# Initialize UserDataDownload object
user_data = UserDataDownload(auth="your_auth_token",
                             username="desired_username")

# Set up client for accessing Twitter API
user_data.set_client(bearer_token)

# Retrieve user data from Twitter
user_data.set_user_data()

# Set the ID of the oldest tweet to retrieve
user_data.set_until_id()

# Set up paginator for retrieving tweets
user_data.set_paginator()

# Create necessary directories for data storage
user_data.set_output_filename()

# Set the maximum number of tweets to retrieve based on rate limit
user_data.set_max_tweets(bearer_token)

# Download tweets and store them
user_data.download()

# Save downloaded tweets
user_data.save()


'''
        

        