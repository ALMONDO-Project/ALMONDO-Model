import os
from downloadData import UserDataDownload
from utils import *

def main():
    BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAAB6rAEAAAAAUETTBU7ohCCUuzTnRu1VHgYw4Vk%3DN5I6Yq9m16CwhIjwHsFrYx87qsqBeHxwD1lA6bksneT5IlsIvS'
        
    # # Define the username of the user whose data you want to download
    username = '@IKEA'

    # # Initialize UserDataDownload object
    user_data = UserDataDownload(username=username)

    # # Set up client for accessing Twitter API
    user_data.set_client(bearer_token=BEARER_TOKEN)

    # # Retrieve user data from Twitter
    user_data.set_user_data()
    
    user_data.set_output_filename()

    # # Set the ID of the oldest tweet to retrieve
    user_data.set_until_id()

    # # Set up paginator for retrieving tweets
    user_data.set_paginator()

    # # Set the maximum number of tweets to retrieve based on rate limit
    user_data.set_max_tweets(bearer_token=BEARER_TOKEN, n=5)

    # # Download tweets and store them
    user_data.download()
    
    user_data.mergedata()

    # Save downloaded tweets
    # user_data.save()
    
    print('>>> process ended')
    
    
    
    
    
if __name__ == "__main__":
    main()