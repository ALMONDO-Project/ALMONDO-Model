import os
import tweepy
import json
import requests

# client = tweepy.client(BEARER_TOKEN)

def users_done_update(log_data_path: str, user: str):
    with open(log_data_path+'/users_done.txt', 'a+'):
        log_data_path.write('\n')
        log_data_path.write(user)
    return

def users_to_do_update(input_data_path: str, log_data_path: str):
    #update the file with remaining users to retreive tweets from
    with open(log_data_path+'/users_done.txt', 'r') as lfile:
        lines = lfile.readlines()
        users_done = [line.strip() for line in lines]
    
    with open(input_data_path+'/users_to_do.txt', 'r') as ifile:
        lines = ifile.readlines() 
        users_to_do = [line.strip() for line in lines]
    
    res = filter(lambda i: i not in users_done, users_to_do)
    
    with open(input_data_path+'/users_to_do.txt', 'w') as ofile:
        users_to_do_string = '\n'.join(res)
        ofile.write(users_to_do_string)
    
    return 

def read_users(input_data_path: str):
    try:
        with open(f'{input_data_path}/users_to_do.txt', 'r') as ifile:
            lines = ifile.readlines() 
            users_to_do = [line.strip() for line in lines]
    except FileNotFoundError:
        users_to_do = []
    return users_to_do

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
        max_tweets_left = int(responsedict['data']['project_cap']) - int(responsedict['data']['project_usage']) - 1000
        return max_tweets_left
    else:
        print("Error:", response.status_code)
        
def get_tweets(client, log_data_path: str, username: str):
    with open(log_data_path+'logfile.txt', 'a') as file:
        print(f"retrieving tweets from {username} timeline")
        file.write(f"START {username}: retrieving tweets from {username} timeline\n")
        user_id = client.get_user(username=username).data.id
        responses = tweepy.Paginator(client.get_users_tweets, user_id, start_date = datetime(year=2023, month=1, day=1, hour = 0, minute = 0), end_date = datetime(year=2023, month=12, day=31, hour=23, minute=59), max_results=1, limit=1)
        counter = 0
        tweet_list = []
        for response in responses:
            counter += 1
            print(f"==> processing {counter * 100} to {(counter + 1) * 100} of {username}'s tweets")
            file.write(f"==> processing {counter * 100} to {(counter + 1) * 100} of {username}'s tweets")
            try:
                for tweet in response.data:  # see any individual tweet by id at: twitter.com/anyuser/status/TWEET_ID_HERE
                    print(tweet)
                    tweet_list.append(dict(tweet))
            except Exception as e:
                print(e)
        file.write(f"END {username}: finished retrieving tweets from {username} timeline\n")
    return tweet_list

def get_oldest_tweet_id(path):
    if not os.path.exists(path):
        return None
    else:
        with open(path, 'r') as file:
            last_tweet = list(json.load(file))[-1]
        return last_tweet['id']


    
    
        
        
    
    


    









    


