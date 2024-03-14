from searchtweets import collect_results, load_credentials, gen_request_parameters
from datetime import date, timedelta
import pandas as pd
import json as serializer
import requests
import os
from os.path import join as join, dirname
import smtplib, ssl
from dotenv import load_dotenv

# Get today's date
today = date.today()


base = "/home/paul/twitter/Twitter/"

class DataDownloader():
    def __init__(self, query, day=6, tweet_fields='text,id,attachments,created_at,lang,author_id,entities,geo', expansions='attachments.media_keys,geo.place_id', media_fields='media_key,type,url,variants,preview_image_url'):
        # If i only want the tweet text i can use tweet_fields='' or tweet_fields=None
        # Both will send no tweet fields, and you'll receive only the default fields (id, text, edit_history_tweet_ids)
        if tweet_fields == '':
            tweet_fields = None

        self.day_downloded = day
        self.filename = f"{str(today-timedelta(day))}"
        # Filename for the day
        
        self.path = f"{base}data/"
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        self.tweets_path = f"{self.path}tweets/"
        if not self.tweets_path:
            self.tweets_path = f"{self.path}tweets/"
        
        if not os.path.exists(self.tweets_path):
            os.makedirs(self.tweets_path)
            
        self.search_credentials = load_credentials(filename=f"{base}cred.yaml", 
                                        yaml_key="search_tweets_cred")
        self.counts_credentials = load_credentials(filename=f"{base}cred.yaml", 
                                        yaml_key="counts_tweets_cred")
        self.search_rule = gen_request_parameters(query,
                                        results_per_call=100,
                                        tweet_fields=tweet_fields,
                                        media_fields=media_fields,
                                        expansions=expansions,
                                        start_time=(today - timedelta(day)).strftime("%Y-%m-%d"),
                                        end_time=(today - timedelta(day-1)).strftime("%Y-%m-%d"))
    
        self.counts_rule = gen_request_parameters(query, granularity="day")

    def update_api_limits(self, day_downloded):
        print('Updating api limits: ')
        def check_usage():
            cred = load_credentials(filename=f"{base}cred.yaml", yaml_key="usage_cred")
            
            def bearer_oauth(r):
                """
                Method required by bearer token authentication.
                """

                r.headers["Authorization"] = f"Bearer {cred['bearer_token']}"
                r.headers["User-Agent"] = "v2UsageTweetsPython"
                return r
            
            response = requests.request('GET', url=cred['endpoint'], auth=bearer_oauth)
            if response.status_code != 200:
                raise Exception(
                    "Request returned an error: {} {}".format(
                        response.status_code, response.text
                    )
                )
            return response.json()
        
        usage = check_usage()['data']
        # Check the api limits
        counts = collect_results(self.counts_rule, result_stream_args=self.counts_credentials)
        if os.path.exists(f"{self.path}limits.json"):
            with open(f"{self.path}limits.json", "r") as f:
                data = serializer.load(f)
        else:
            data = {}    
            data['Started: '] = str(today) 
            data['Remaining_Month_Cap_Limit'] = int(int(usage['project_cap']) - int(usage['project_usage']))
            
        data['Month_Cap_Limit'] = int(usage['project_cap'])
        data['Used_Requests'] = int(usage['project_usage']) + int(counts[0]['data'][0]['tweet_count'])
        data['Cap_Reset_Day'] = int(usage['cap_reset_day'])
        
        data['Remaining_Month_Cap_Limit'] = int(data['Month_Cap_Limit']) - int(data['Used_Requests'])
        
        if today.day > data['Cap_Reset_Day']:
            data['Cap_Remaining_Days'] = int((date(today.year, today.month+1, int(data['Cap_Reset_Day'])) - today).days)
        else:
            data['Cap_Remaining_Days'] = int((date(today.year, today.month, int(data['Cap_Reset_Day'])) - today).days)
        
        data['Day_Cap_Limit'] = int(int(data['Remaining_Month_Cap_Limit']) / int(data['Cap_Remaining_Days']))
        
        self.max_tweets_per_day = int(data['Day_Cap_Limit'])
        # Print the limits
        print('#########################################################')
        print(data)
        print('#########################################################')
        # Save the limits
        
        day_donwloading = (today - timedelta(day_downloded)).strftime("%Y-%m-%d")
        serializer.dump(data, open(f"{self.path}limits.json", "w"))
        serializer.dump(counts, open(f"{self.path}counts/{str(today)}_counts_{day_donwloading}.json", "w"))
    
    def join_and_save(self):
        all_tweets_json = []
        for file in os.listdir(f"{self.path}tweets/"):
            if file.endswith(".json"):
                file = serializer.loads(open(f"{self.path}tweets/{file}").read())
                all_tweets_json.append(file)
        all_tweets_json = [tweet for tweets in all_tweets_json for tweet in tweets]
        serializer.dump(all_tweets_json, open(f"{self.path}all_tweets.json", "w"))

    def download_and_save(self):
        self.update_api_limits(self.day_downloded)
        # Collecting tweets
        print(f"Collecting tweets for day {self.filename}")
        tweets = collect_results(self.search_rule, max_tweets=self.max_tweets_per_day, result_stream_args=self.search_credentials)
        print(f"Dumping data...")
        serializer.dump(tweets, open(f"{self.tweets_path}{self.filename}.json", "w"))
        try:
            self.join_and_save()
        except:
            print("Can't join the data")
            pass
        print(f"Done.\n")
        

        
downloader = DataDownloader(day=1
                            , query="lampedusa -is:retweet")
downloader.download_and_save()