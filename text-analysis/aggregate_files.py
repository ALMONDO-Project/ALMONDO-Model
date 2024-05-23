import os
import json
import pandas as pd
from datetime import datetime

def process_json_file(file_path, folder_name):
    print(f'processing data for user: {folder_name}')
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.decoder.JSONDecodeError:
            print(f"Error reading file: {file_path}")
            return []
    tweets = []
    for _, tweet_data in data.items():
        created_at = tweet_data['created_at']
        if '2023-01-01' <= created_at[:10] <= '2023-12-31':
            tweet_data['username'] = folder_name.lower()
            tweets.append(tweet_data)
        else:
            print(f'tweet created at {created_at} for user {folder_name}')
    return tweets

def traverse_folders(path):
    valid_tweets_list = []
    c=0
    for root, _, files in os.walk(path):
        folder_name = os.path.basename(root)
        c+=1
        for file in files:
            if file.endswith('.json') and not file.startswith('page'):
                file_path = os.path.join(root, file)
                valid_tweets_list.extend(process_json_file(file_path, folder_name))
    print(c)
    return valid_tweets_list

def save_dataset(dataset, output_file, file_format='json'):
    if file_format.lower() == 'json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
            print('Json file created')
    elif file_format.lower() == 'csv':
        df = pd.DataFrame(dataset)
        df.to_csv(output_file, index=False)
    else:
        print("Unsupported file format. Please choose either 'json' or 'csv'.")

# Main function
if __name__ == "__main__":
    path = '../data/log/'  # Your path here
    output_file = '../data/out/filtered_tweets_3.json'  # Output file name
    tweets = traverse_folders(path)
    save_dataset(tweets, output_file, file_format='json')
    print('Dataset saved')