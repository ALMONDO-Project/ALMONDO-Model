#import statements
import os
import json
import tweepy
import tqdm
import math
from json.decoder import JSONDecodeError
from utils import *

#class definition

BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAAB6rAEAAAAAUETTBU7ohCCUuzTnRu1VHgYw4Vk%3DN5I6Yq9m16CwhIjwHsFrYx87qsqBeHxwD1lA6bksneT5IlsIvS'

class UserDataDownload():
    def __init__(self,
                 bearer_token = 'AAAAAAAAAAAAAAAAAAAAAAB6rAEAAAAAUETTBU7ohCCUuzTnRu1VHgYw4Vk%3DN5I6Yq9m16CwhIjwHsFrYx87qsqBeHxwD1lA6bksneT5IlsIvS',
                 username='',
                 expansions='attachments.media_keys,geo.place_id,author_id',
                 tweet_fields='text,id,attachments,created_at,lang,author_id,entities,geo,public_metrics',
                 media_fields='media_key,type,url,variants,preview_image_url',
                 user_fields='id,username,description,public_metrics,verified',
                 until_id = None):
        # Initialize UserDataDownload object with required parameters and default values
        self.username = username.replace('@', '')  # Remove '@' from username if present
        self.tweet_fields = tweet_fields.split(',')  # Split tweet fields into list
        self.media_fields = media_fields.split(',')  # Split media fields into list
        self.expansions = expansions.split(',')  # Split expansions into list
        self.user_fields = user_fields.split(',')  # Split user fields into list
        self.until_id = until_id  # Set until_id for pagination
        self.tweets = {}  # Dictionary to store tweets
        
    def set_client(self, bearer_token, wait_on_rate_limit=True):
        # Set Twitter API client with specified parameters
        self.client = tweepy.Client(bearer_token, 
                                    # expansions = self.expansions,
                                    # media_fields = self.media_fields,
                                    # tweet_fields = self.tweet_fields,
                                    # user_fields = self.user_fields,
                                    # return_type=dict,
                                    wait_on_rate_limit=wait_on_rate_limit)
        
    def set_user_data(self):
        # Get user data using Twitter API client
        self.user = self.client.get_user(username=self.username)
        self.user_id = self.user.data.id
        
    def set_until_id(self):
        print(f'retrieving tweets with id older than {self.until_id}') #the starting tweet id should be less than this
        if os.path.exists(self.userlogpath):
            try:
                json_files = [file for file in os.listdir(self.userlogpath) if file.endswith('.json')]
                json_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.userlogpath, x)), reverse=True)
                less_recent_tweet_id = json_files[0].replace('.json', '')
            except IndexError:
                less_recent_tweet_id = None
            print(f'retrieving tweets with id older than {self.until_id}') #the starting tweet id should be less than this
        else:
            less_recent_tweet_id = None
            print('>>> no saved data for this user')
        self.until_id = less_recent_tweet_id
            

        
    def set_paginator(self):
        # Initialize paginator for fetching user tweets


        print(f'{self.paginator} \n object was created')
    
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
        self.filename = f'{self.useroutdatapath}/{self.username}_tweets.json'
            
    # def set_max_tweets(self, bearer_token, n=None):
    #     # Set the maximum number of tweets to download
    #     left = compute_max_tweets(bearer_token)  # Compute remaining tweets allowed
    #     print(f'>>> there are {left} tweets left to download')
    #     self.max_tweets = min(left, n) if n is not None else left  # Set max_tweets to minimum of remaining tweets and n
    #     print(f'Asking to download {self.max_tweets} tweets at max')
    #     if left <= 0:
    #         print('no tweets left')
    #         return None
        
    # def create_tweet_data_list(self):
    #     self.tweets = []
    #     if os.path.exists(self.userlogpath):
    #         try:
    #             json_files = [file for file in os.listdir(self.userlogpath) if file.endswith('.json')]
    #             json_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.userlogpath, x)), reverse=True)
    #             for json_file in json_files:
    #                 with open(json_file, 'r') as file:
    #                     tweet = json.load(file)
    #                     self.tweets.append(tweet)
    #         except FileNotFoundError:
    #             pass         
            
    def dump_tweets(self, tweet, tweet_data):
        with open(f'data/log/{self.username}/{tweet.id}.json', 'w') as file:
            json.dump(tweet_data, file, indent=4, sort_keys=True, default=str)  # Write tweet data to file
            
    def update_users_done(self):
        print(f'>>> finished with user {self.username}')
        with open(f'data/log/users_done.txt', 'r') as file:
            file.write(f'{self.username}\n')

    def download(self):
        count = compute_max_tweets(self.bearer_token)
        print(f'>>> the maximum number of tweets i can retrieve is {count}')
        
        max_results = 100
        limit = math.ceil(count / max_results)
        
        print(f'>>> the process will do at most {limit} calls asking for at most {max_results} tweets per call')
        
        self.paginator = tweepy.Paginator(self.client.get_users_tweets,
                                    self.user_id,
                                    exclude = ['retweets'],
                                    until_id = self.until_id,
                                    expansions = self.expansions,
                                    tweet_fields = self.tweet_fields,
                                    media_fields = self.media_fields,
                                    user_fields = self.user_fields,
                                    limit = limit,
                                    max_results = max_results)
        
        print('>>> started retrieving tweets')
        for page in tqdm.tqdm(self.paginator):
            
            next_token = page.meta["next_token"] #non l'ho provata sta riga di codice non so se funziona
            for tweet in page.data: #così limit = inf però comuque non dovrebbe scaricarmi più di max_results però mi sembra che vada avanti a oltranza senza badare a quel parametro boh
                tweet_data = {tweet.data['id']: tweet.data}
                self.dump_tweets(tweet, tweet_data)
                count += 1
                if count > self.max_results:
                    return
                
            if not next_token:
                self.update_users_done()
                print(f'>>> {count} total tweets downloaded in this session')        
                return     
        
    def save(self):
        if len(self.tweets) > 0:
            # Save tweets data to file
            with open(f'{self.filename}', 'w') as file:
                json.dump(self.tweets, file, indent=4, sort_keys=True, default=str)
        else:
            print('there were no tweets to save')
    
    def mergedata(self):
        # Directory containing the JSON files
        directory = self.userlogpath

        # Initialize an empty dictionary to store merged data
        merged_data = {}

        # Iterate over each JSON file in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r') as file:
                    data = json.load(file)
                    # Extract the tweet ID and its corresponding data
                    tweet_id = list(data.keys())[0]
                    tweet_data = data[tweet_id]
                    # Store the data in the merged dictionary
                    merged_data[tweet_id] = tweet_data

        # Write the merged data into a new JSON file
        output_file = f'{self.useroutdatapath}/{self.username}_tweets_merged.json'
        with open(output_file, 'w') as outfile:
            json.dump(merged_data, outfile, indent=4)

        print("Merged JSON file created successfully:", output_file)
            
    def get_tweets(self):
        return self.tweets

    def get_paginator(self):
        return self.paginator


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
        

        