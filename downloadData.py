import os
import json
from datetime import datetime
import logging
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

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Log to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Log to file
        fh = logging.FileHandler(f"{self.home}/data/logs/{username}_log.txt")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def make_dirs(self):
        try:
            if not os.path.exists(self.tweets_path):
                os.makedirs(self.tweets_path)
                self.logger.info(f'Created folder: {self.tweets_path}')
            else:
                self.logger.info(f'{self.tweets_path} already exists')
        except OSError as e:
            self.logger.error(f"Error creating directory: {self.tweets_path} - {e}")

    def set_max_id(self):
        try:
            self.max_id = get_oldest_tweet_id(self.tweets_path)
            if not self.max_id:
                raise FileNotFoundError("Empty file exists")
            self.logger.info(f'Oldest tweet downloaded is {self.max_id} at {self.tweets_path}')
        except FileNotFoundError:
            self.max_id = None
            self.logger.info('No previous tweets downloaded, setting max_id to None')

    # Define other methods similarly

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

            self.logger.info(f"Collecting tweets for user {self.username}")

            tweets = collect_results(self.search_rule, max_tweets=self.max_tweets, result_stream_args=self.search_credentials)
            self.tweets = tweets

            # Save tweets to file with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename_with_timestamp = f"{self.home}/{self.username}_tweets_{timestamp}.json"
            with open(filename_with_timestamp, 'w') as file:
                json.dump(self.tweets, file)
        except Exception as e:
            self.logger.error(f"Error downloading tweets: {e}")

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
            self.logger.error(f"Error saving tweets: {e}")
