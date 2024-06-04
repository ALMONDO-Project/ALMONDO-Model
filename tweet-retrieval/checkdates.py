from downloadData import UserDataDownload
import math
from utils import *

def read_users(path):
    with open(f'{path}/users_data.json', 'r') as file:
        users_data = json.load(file)
    return users_data

def users_update(users_data, username, path):
    users_data['to_do'].remove(username)
    users_data['done'].append(username)
    with open(f'{path}/users_data.json', 'w') as file:
        json.dump(users_data, file)
    return users_data

def main():
    users_data = read_users('../data')
    usernames = users_data['all']
    # Dictionary to store the dates for each username
    dates = {}
    # Iterate over each username
    for username in usernames:
        tweet_ids = []
        username = username.replace('@', '')
        # List all files in the directory for the given username
        user_dir = f'../data/log/{username}/'
        if os.path.exists(user_dir):
            dates[username] = {}
            for filename in os.listdir(user_dir):
                if filename.endswith('.json') and not filename.startswith('page'):
                    # Extract tweet_id from the filename
                    tweet_id = filename.split('.')[0]
                    tweet_ids.append(tweet_id)
                
            min_id = min(tweet_ids)
            max_id = max(tweet_ids)
            
            try:
                with open(f'../data/log/{username}/{min_id}.json') as file:
                    data = json.load(file)
                dates[username]['min_date'] = data[min_id]['created_at']
                with open(f'../data/log/{username}/{max_id}.json') as file:
                    data = json.load(file)
                dates[username]['max_date'] = data[max_id]['created_at']
            except:
                continue
                
                    
    # Save the dates dictionary to a JSON file
    output_file = '../data/log/dates.json'
    with open(output_file, 'w') as file:
        json.dump(dates, file, indent=4)                    
                    
if __name__ == "__main__":
    main()