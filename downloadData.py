#import statements
import os
import json
import tweepy
import tqdm
import math
import pickle
import time
from json.decoder import JSONDecodeError
from utils import *

class TweetAlreadyDumpedException():
    pass

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
        self.bearer_token = bearer_token
        self.username = username.replace('@', '')  # Remove '@' from username if present
        self.tweet_fields = tweet_fields.split(',')  # Split tweet fields into list
        self.media_fields = media_fields.split(',')  # Split media fields into list
        self.expansions = expansions.split(',')  # Split expansions into list
        self.user_fields = user_fields.split(',')  # Split user fields into list
        self.until_id = until_id  # Set until_id for pagination
        print(f'>>> class initialized for user {self.username}')
        print(f'>>> initial parameters are:')
        print(f'>>> tweet fields = {self.tweet_fields}')
        print(f'>>> media_fields = {self.media_fields}')
        print(f'>>> user_fields = {self.user_fields}')
        print(f'>>> expansions = {self.expansions}')
        print(f'>>> until_id = {self.until_id}')
        
    def set_client(self, wait_on_rate_limit=True):
        # Set Twitter API client with specified parameters
        self.client = tweepy.Client(self.bearer_token, 
                                    wait_on_rate_limit=wait_on_rate_limit)
        print('>>> client initialized')
        
    def set_user_data(self):
        # Get user data using Twitter API client
        self.user = self.client.get_user(username=self.username)
        self.user_id = self.user.data.id
        
    # def set_until_id(self):
    #     print(f'retrieving tweets with id older than {self.until_id}') #the starting tweet id should be less than this
    #     if os.path.exists(self.userlogpath):
    #         try:
    #             json_files = [file.replace('.json', '') for file in os.listdir(self.userlogpath) if file.endswith('.json')]
    #             json_files.sort()
    #             smallest_id = json_files[0].replace('.json', '')
    #         except IndexError:
    #             smallest_id = None
    #     else:
    #         smallest_id = None
    #         print(f'>>> no saved data for user {self.username}')
        
    #     self.until_id = smallest_id
    #     print(f'>>> retrieving tweets with id older than {self.until_id}') #the starting tweet id should be less than this
              
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
            print('>>> necessary directories created')
        except OSError as e:
            print(e)
            
    # def set_output_filename(self):
    #     # Set output filename for storing tweets
    #     self.make_dirs()
    #     self.filename = f'{self.useroutdatapath}/{self.username}_tweets.json'
            
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
        if not os.path.exists(f'data/log/{self.username}/{tweet.id}.json'):
            with open(f'data/log/{self.username}/{tweet.id}.json', 'w') as file:
                json.dump(tweet_data, file, indent=4, sort_keys=True, default=str)  # Write tweet data to file
        else:
            print(f'>>> {tweet.id} already dumped, something went wrong')
            raise TweetAlreadyDumpedException
            exit()   
            
    def set_time_limits(self):
        self.start_time = "2023-01-01T00:00:00.000Z"
        
        if is_before(new_end_time(self.userlogpath), "2023-12-31T23:59:59.000Z"):
            self.end_time new_end_time(self.userlogpath)
        else:
            self.end_time= "2023-12-31T23:59:59.000Z"
        
        if not is_before(self.start_time, self.end_time):
            raise Exception
        
            
           
        
    def download(self, count):
        print(f'>>> the maximum number of tweets i can retrieve is {count}')
        
        max_results = 100
        limit = math.ceil(count / max_results)
        
        print(f'>>> the process will do at most {limit} calls asking for at most {max_results} tweets per call')
        
        self.set_time_limits()
        
        self.paginator = tweepy.Paginator(self.client.get_users_tweets,
                                    self.user_id,
                                    exclude = ['retweets'],
                                    expansions = self.expansions,
                                    tweet_fields = self.tweet_fields,
                                    media_fields = self.media_fields,
                                    user_fields = self.user_fields,
                                    start_time = self.start_time,
                                    end_time = self.end_time,
                                    limit = limit,
                                    max_results = max_results)
        
        print(f'>>> started retrieving tweets from {self.start_time} to {self.end_time}')
        
        for page in self.paginator:
            print(">>> starting a new request") 
            try:
                next_token = page.meta["next_token"] #non l'ho provata sta riga di codice non so se funziona
            except KeyError:
                next_token = None
                
            with open(f'data/log/{self.username}/last_page.pickle', 'wb') as file:
                pickle.dump(page, file)
            
            try:    
                for tweet in tqdm.tqdm(page.data): #così limit = inf però comuque non dovrebbe scaricarmi più di max_results però mi sembra che vada avanti a oltranza senza badare a quel parametro boh
                    tweet_data = {tweet.data['id']: tweet.data}
                    self.dump_tweets(tweet, tweet_data)
                    count -= 1
                    if count <= 0:
                        return
            except TypeError:
                with open('data/log/users_done.txt', 'a') as file:
                    file.write('@'+self.username)
                    file.write('\n')
                return 
            except TweetAlreadyDumpedException:
                print('>>> check if unitll_id is set properly')
                return
               
            if not next_token:
                with open('data/log/users_done.txt', 'a') as file:
                    file.write('@'+self.username)
                    file.write('\n')
                return     
            
            print('>>> going to sleep for 3 minutes')
            time.sleep(2 * 60) #in this way it does a request every two minutes so it does 7/8 requests every 15 minutes
        
    # def save(self):
    #     if len(self.tweets) > 0:
    #         # Save tweets data to file
    #         with open(f'{self.filename}', 'w') as file:
    #             json.dump(self.tweets, file, indent=4, sort_keys=True, default=str)
    #     else:
    #         print('there were no tweets to save')
    
    # def mergedata(self):
    #     # Directory containing the JSON files
    #     directory = self.userlogpath

    #     # Initialize an empty dictionary to store merged data
    #     merged_data = {}

    #     # Iterate over each JSON file in the directory
    #     for filename in os.listdir(directory):
    #         if filename.endswith('.json'):
    #             filepath = os.path.join(directory, filename)
    #             with open(filepath, 'r') as file:
    #                 data = json.load(file)
    #                 # Extract the tweet ID and its corresponding data
    #                 tweet_id = list(data.keys())[0]
    #                 tweet_data = data[tweet_id]
    #                 # Store the data in the merged dictionary
    #                 merged_data[tweet_id] = tweet_data

    #     # Write the merged data into a new JSON file
    #     output_file = f'{self.useroutdatapath}/{self.username}_tweets_merged.json'
    #     with open(output_file, 'w') as outfile:
    #         json.dump(merged_data, outfile, indent=4)

    #     print("Merged JSON file created successfully:", output_file)
            
    # def get_tweets(self):
    #     return self.tweets

    # def get_paginator(self):
    #     return self.paginator
