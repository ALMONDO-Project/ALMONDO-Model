import os
import json
from datetime import datetime
from searchtweets import collect_results, load_credentials, gen_request_parameters
from utils import compute_max_tweets, get_oldest_tweet_id

class UserDataDownloader:
    def __init__(self,
                 username,
                 start_time="2023-01-01",
                 end_time="2023-12-31",
                 tweet_fields='text,id,attachments,created_at,lang,author_id,entities,geo',
                 expansions='attachments.media_keys,geo.place_id',
                 media_fields='media_key,type,url,variants,preview_image_url',
                 results_per_call=100, 
                 home = '.'):

        self.username = username
        self.start_time = start_time
        self.end_time = end_time
        self.tweet_fields = tweet_fields
        self.media_fields = media_fields
        self.expansions = expansions
        self.results_per_call = results_per_call
        
        self.max_id = None
        self.tweets = {}
        self.home = home
        self.tweets_path = f"{self.home}/data/out/{username}"
        self.filename = f"{self.tweets_path}/{username}_tweets.json"
        self.search_credentials = load_credentials(filename=f"{self.home}/cred.yaml", yaml_key="search_tweets_cred")
        self.query = f"from:{username} -is:retweet"

    def _handle_error(self, message):
        print(f"Error: {message}")

    def make_dirs(self):
        try:
            if not os.path.exists(self.tweets_path):
                os.makedirs(self.tweets_path)
                print(f'Created folder: {self.tweets_path}')
            else:
                print(f'{self.tweets_path} already exists')
        except OSError as e:
            self._handle_error(f"Error creating directory: {self.tweets_path} - {e}")

    def set_max_id(self):
        self.max_id = get_oldest_tweet_id(self.filename) #o è none o è qualcosa 

    def set_max_tweets(self, BEARER_TOKEN, n=None):
        left = compute_max_tweets(BEARER_TOKEN) - 1000
        print(f'>>> there are {left} tweets left to download')
        self.max_tweets = min(left, n) if n is not None else left
        print(f'Asking to download {self.max_tweets} tweets at max')
        if left <= 0:
            print('no tweets left')
            return None


    def download(self):
        self.search_rule = gen_request_parameters(self.query,
                                                    granularity=None,
                                                    results_per_call=self.results_per_call,
                                                    tweet_fields=self.tweet_fields,
                                                    media_fields=self.media_fields,
                                                    expansions=self.expansions,
                                                    until_id=self.max_id,
                                                    start_time=self.start_time,
                                                    end_time=self.end_time)

        print(f"Collecting tweets for user {self.username}")

        self.tweets = collect_results(self.search_rule, max_tweets=self.max_tweets, result_stream_args=self.search_credentials)

        # Save tweets to file with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename_with_timestamp = f"{self.home}/{self.username}_tweets_{timestamp}.json"
        with open(filename_with_timestamp, 'w') as file:
            json.dump(self.tweets, file)

    def save_tweets(self):
        if self.tweets:
            try:
                with open(self.filename, 'r') as ifile:
                    tweet_dict = json.load(ifile)
                    progressive_tweet_id = max(map(int, tweet_dict.keys())) + 1
            except (FileNotFoundError, json.JSONDecodeError):
                tweet_dict = {}
                progressive_tweet_id = 0

            for tweet in self.tweets:
                tweet_dict[str(progressive_tweet_id)] = dict(tweet)
                progressive_tweet_id += 1

            with open(self.filename, 'w') as ofile:
                json.dump(tweet_dict, ofile)
                
            return self.tweets 
        
        else:
            print('no tweets to save')
            
            return None
            
