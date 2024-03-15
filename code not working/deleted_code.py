def main():
    BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAAB6rAEAAAAAUETTBU7ohCCUuzTnRu1VHgYw4Vk%3DN5I6Yq9m16CwhIjwHsFrYx87qsqBeHxwD1lA6bksneT5IlsIvS'
    input_data_path = 'data/in/'
    output_data_path = 'data/out/'
    log_data_path = 'data/log/'
    
    # users_to_do_update(input_data_path, log_data_path)
        
    client = tweepy.Client(BEARER_TOKEN, wait_on_rate_limit=True)

    users = read_users(input_data_path)

    tweet_available_count = compute_max_tweets(BEARER_TOKEN)

    usernames = users_to_collect_from(client, users, tweet_available_count)
    
    # for username in usernames:
    #     username = username.replace('@', '')
    #     tweet_list = get_tweets(client, user)
    #     write_results(tweet_list, username, out_data_path)
    #     users_done_update(log_data_path, [username])
       
       
### main
        
if __name__ == '__main__':
    main()
    
### end of main