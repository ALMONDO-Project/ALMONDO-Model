import os
from downloadData import UserDataDownload
from utils import *

def main():
    BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAAB6rAEAAAAAUETTBU7ohCCUuzTnRu1VHgYw4Vk%3DN5I6Yq9m16CwhIjwHsFrYx87qsqBeHxwD1lA6bksneT5IlsIvS'
    
    home = 'C:/Users/valen/Github/almondo-tweets-retrieval'
    
    # Define the username of the user whose data you want to download
    username = 'desired_username'

    # Initialize UserDataDownload object
    user_data = UserDataDownload(username=username)

    try:
        # Set up client for accessing Twitter API
        user_data.set_client(bearer_token=BEARER_TOKEN)

        # Retrieve user data from Twitter
        user_data.set_user_data()

        # Set the ID of the oldest tweet to retrieve
        user_data.set_until_id()

        # Set up paginator for retrieving tweets
        user_data.set_paginator()

        # Create necessary directories for data storage
        user_data.set_output_filename()

        # Set the maximum number of tweets to retrieve based on rate limit
        user_data.set_max_tweets(bearer_token=BEARER_TOKEN)

        # Download tweets and store them
        tweets = user_data.download()

        # Save downloaded tweets
        user_data.save()

        print("Download completed successfully!")
    
    except Exception as e:
        
        print(f"Error occurred: {e}")

    if tweets:
        print(f'>>> process ended for user {username}')
        
        print(f'>>> saving logs')
        
        users_done_update(f'{home}/data/log/', username)
        
        users_to_do_update(f'{home}/data/in/', f'{home}/data/log/')

    print('>>> process ended')
    
    
    
    
    
if __name__ == "__main__":
    main()