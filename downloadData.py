import os
from searchtweets import collect_results, load_credentials, gen_request_parameters
import json
from datetime import datetime
from utils import compute_max_tweets, get_oldest_tweet_id

class NoTweetsLeftException(Exception):
    pass

class NoTweetsToSaveException(Exception):
    pass

class UserDataDownloader:
    def __init__(self,
                 username,
                 start_time="2023-01-01",
                 end_time="2023-12-31",
                 tweet_fields='text,id,attachments,created_at,lang,author_id,entities,geo',
                 expansions='attachments.media_keys,geo.place_id',
                 media_fields='media_key,type,url,variants,preview_image_url',
                 results_per_call=100):

        self.username = username
        self.start_time = start_time
        self.end_time = end_time
        self.tweet_fields = tweet_fields
        self.media_fields = media_fields
        self.expansions = expansions
        self.results_per_call = results_per_call
        
        self.max_id = None
        self.tweets = {}
        self.home = os.getcwd()
        self.tweets_path = f"{self.home}/data/out/{username}"
        self.filename = f"{self.tweets_path}/{username}_tweets.json"
        self.search_credentials = load_credentials(filename=f"{self.home}/cred.yaml", yaml_key="search_tweets_cred")
        self.query = f"from:{username} -is:retweet"

    def make_dirs(self):
        try:
            if not os.path.exists(self.tweets_path):
                os.makedirs(self.tweets_path)
                print(f'Created folder: {self.tweets_path}')
            else:
                print(f'{self.tweets_path} already exists')
        except OSError as e:
            print(f"Error creating directory: {self.tweets_path} - {e}")

    def set_max_id(self):
        try:
            self.max_id = get_oldest_tweet_id(self.tweets_path)
            if not self.max_id:
                raise FileNotFoundError("Empty file exists")
            print(f'Oldest tweet downloaded is {self.max_id} at {self.tweets_path}')
        except FileNotFoundError:
            self.max_id = None
            print('No previous tweets downloaded, setting max_id to None')

    def set_max_tweets(self, BEARER_TOKEN, n=None):
        try:
            left = compute_max_tweets(BEARER_TOKEN) - 1000
            if left > 0:
                self.max_tweets = min(left, n) if n is not None else left
                print(f'Asking to download {self.max_tweets} tweets at max')
            else:
                raise NoTweetsLeftException('No tweets left to download')
        except Exception as e:
            print(f"Error setting max tweets: {e}")

    def download(self):
        try:
            self.search_rule = gen_request_parameters(self.query,
                                                      results_per_call=self.results_per_call,
                                                      tweet_fields=self.tweet_fields,
                                                      media_fields=self.media_fields,
                                                      expansions=self.expansions,
                                                      max_id=self.max_id,
                                                      start_time=self.start_time,
                                                      end_time=self.end_time)

            print(f"Collecting tweets for user {self.username}")

            tweets = collect_results(self.search_rule, max_tweets=self.max_tweets, result_stream_args=self.search_credentials)
            self.tweets = tweets

            # Save tweets to file with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename_with_timestamp = f"{self.home}/{self.username}_tweets_{timestamp}.json"
            with open(filename_with_timestamp, 'w') as file:
                json.dump(self.tweets, file)
        except Exception as e:
            print(f"Error downloading tweets: {e}")

    def save_tweets(self):
        try:
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
            else:
                raise NoTweetsToSaveException('No tweets to save')
        except Exception as e:
            print(f"Error saving tweets: {e}")
