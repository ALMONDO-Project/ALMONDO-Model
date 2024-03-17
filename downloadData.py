#import statements
import os
import json
import tweepy
import tqdm
import math
from json.decoder import JSONDecodeError
from utils import *

#class definition

class UserDataDownload():
    def __init__(self,
                 username='',
                 expansions='attachments.media_keys,geo.place_id,author_id',
                 tweet_fields='text,id,attachments,created_at,lang,author_id,entities,geo,public_metrics',
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
        self.max_id = None  # Initialize max_id for pagination
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
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as file:
                data = json.load(file)
            # Initialize variables to store the most recent tweet ID and its creation date
            less_recent_tweet_id = None
            less_recent_creation_date = None

            # Iterate through the tweets data to find the most recent tweet
            for tweet_dict in data:
                for tweet_id, tweet_info in tweet_dict.items():
                    creation_date = tweet_info['created_at']
                    if less_recent_creation_date is None or creation_date < less_recent_creation_date:
                        less_recent_creation_date = creation_date
                        less_recent_tweet_id = tweet_id
            
            print(f'retrieving tweets with id older than {self.until_id}') #the starting tweet id should be less than this
        
        elif os.path.exists(self.userlogpath):
            json_files = [file for file in os.listdir(self.userlogpath) if file.endswith('.json')]
            json_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.userlogpath, x)), reverse=True)
            less_recent_tweet_id = json_files[0].replace('.json', '')
            print(f'retrieving tweets with id older than {self.until_id}') #the starting tweet id should be less than this
            
        else:
            less_recent_tweet_id = None
            print('>>> no saved data for this user')
            
        self.until_id = less_recent_tweet_id
            

        
    def set_paginator(self):
        # Initialize paginator for fetching user tweets
        print()
        print(f'Asking tweets of {self.username} {self.user_id} from tweet id {self.until_id}.\n \
        Max number of tweets asked is {self.max_tweets}.\n\
        Expansions asked are {self.expansions}\n\
        Tweet fields asked are {self.tweet_fields}\n\
        Media fields asked are {self.media_fields}\n\
        User fields asked are {self.user_fields}')
        
        '''Se apro il codice di .get_user_tweets() mi dice che: 
            max_results : int | None
            Specifies the number of Tweets to try and retrieve, up to a maximum
            of 100 per distinct request. By default, 10 results are returned if
            this parameter is not supplied. The minimum permitted value is 5.
            It is possible to receive less than the ``max_results`` per request
            throughout the pagination process.            
            '''
        self.paginator = tweepy.Paginator(self.client.get_users_tweets,
                                           self.user_id,
                                           exclude = ['retweets'],
                                           until_id = self.until_id,
                                           max_results=self.max_tweets, 
                                           expansions = self.expansions,
                                           tweet_fields = self.tweet_fields,
                                           media_fields = self.media_fields,
                                           user_fields = self.user_fields)
        print()
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
        if os.path.exists(f'{self.filename}'):
            with open(f'{self.filename}', 'r') as file:
                self.tweets = list(json.load(file))
        else:
            self.tweets = []
        print('>>> started retrieving tweets')
        for page in tqdm.tqdm(self.paginator):
            pagination_token = page.meta["next_token"] #non l'ho provata sta riga di codice non so se funziona
            for tweet in page.data:#così limit = inf però comuque non dovrebbe scaricarmi più di max_results però mi sembra che vada avanti a oltranza senza badare a quel parametro boh
                tweet_data = {tweet.data['id']: tweet.data}
                self.tweets.append(tweet_data)
                with open(f'data/log/{self.username}/{tweet.id}.json', 'w') as file:
                    json.dump(tweet_data, file, indent=4, sort_keys=True, default=str)  # Write tweet data to file
            if not pagination_token:
                print('>>> finished with this user')
                with open(f'data/log/users_done.txt', 'r') as file:
                    file.write(f'{self.username}\n')
                break
        count=len(self.tweets)
        print(f'>>> {count} tweets downloaded')        
        
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
        

        