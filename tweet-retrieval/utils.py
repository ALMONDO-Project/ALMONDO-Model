import os
import tweepy
import json
import requests
from datetime import datetime, timedelta


# client = tweepy.client(BEARER_TOKEN)
def write_on_file(what, where):
    with open(where, 'w') as file:
        file.write(what)
    
def compute_max_tweets(bearer_token: str):
    url = 'https://api.twitter.com/2/usage/tweets'
    headers = {
        'Authorization': f'Bearer {bearer_token}'
    }

    response = requests.get(url, headers=headers)
    
    print(response)

    if response.status_code == 200:
        responsedict = response.json()
        # Process the response data here
        max_tweets_left = int(responsedict['data']['project_cap']) - int(responsedict['data']['project_usage'])
        return max_tweets_left
    else:
        print("Error:", response.status_code)
        
def log_message(message):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    LOG_FILE_PATH = "../data/log/messages.log"
    with open(LOG_FILE_PATH, "a+") as log_file:
        log_file.write(str(current_time) + "    " + message + "\n")



    









    


