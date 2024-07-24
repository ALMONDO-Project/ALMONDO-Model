import pandas as pd
import json
from tqdm.notebook import tqdm

# Function to extract tags
def extract_tags(tag_list):
    if type(tag_list) == list:
        return [entry['tag'] for entry in tag_list]
    else:
        return None

# Load the JSON file into a DataFrame
with open('../data/out/filtered_tweets_5.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Create a DataFrame from the JSON data
df = pd.DataFrame(data)  # Transpose to set tweets as rows

public_metrics_expanded = pd.json_normalize(df['public_metrics'])
df = df.join(public_metrics_expanded)
public_metrics_expanded = pd.json_normalize(df['entities'])
df = df.join(public_metrics_expanded)
public_metrics_expanded = pd.json_normalize(df['attachments'])
df = df.join(public_metrics_expanded)
df['hashtags'] = df['hashtags'].apply(extract_tags)


df.to_csv('data/data_3.csv')