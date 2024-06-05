import pandas as pd
import json
from tqdm.notebook import tqdm

# Load the JSON file into a DataFrame
with open('../data/out/filtered_tweets_4.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Create a DataFrame from the JSON data
df = pd.DataFrame(data)  # Transpose to set tweets as rows

with tqdm(total=len(df)) as pbar:
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        attachments = row['attachments']
        if type(attachments) == dict:
            for key, value in attachments.items():
                # Check if column with key name already exists
                if key in df.columns:
                    # Update existing column value
                    df.at[index, f'attachments_{key}'] = value
                else:
                    # Create new column and fill with NaN
                    df[f'attachments_{key}'] = pd.Series([value if i == index else None for i in range(len(df))])
        entities = row['entities']
        if type(entities) == dict:
            for key, value in entities.items():
                if key in df.columns:
                    # Update existing column value
                    df.at[index, f'entities_{key}'] = value
                else:
                    # Create new column and fill with NaN
                    df[f'entities_{key}'] = pd.Series([value if i == index else None for i in range(len(df))])
        public_metrics = row['public_metrics']
        if type(public_metrics) == dict:
            for key, value in public_metrics.items():
                if key in df.columns:
                    # Update existing column value
                    df.at[index, f'public_metrics_{key}'] = value
                else:
                    # Create new column and fill with NaN
                    df[f'public_metrics_{key}'] = pd.Series([value if i == index else None for i in range(len(df))])   
        pbar.update(1)  

with tqdm(total=len(df)) as pbar:
    for index, row in df.iterrows():
        annotations = row['entities_annotations']
        if annotations:
            for d in annotations:
                for key, value in d.items():
                    # Check if column with key name already exists
                    if key in df.columns:
                        # Update existing column value
                        df.at[index, f'annotations_{key}'] = value
                    else:
                        # Create new column and fill with NaN
                        df[f'annotations_{key}'] = pd.Series([value if i == index else None for i in range(len(df))])
        pbar.update(1)

with tqdm(total=len(df)) as pbar:
    for index, row in df.iterrows():
        annotations = row['entities_urls']
        if annotations:
            for d in annotations:
                for key, value in d.items():
                    # Check if column with key name already exists
                    if key in df.columns:
                        # Update existing column value
                        df.at[index, f'urls_{key}'] = value
                    else:
                        # Create new column and fill with NaN
                        df[f'urls_{key}'] = pd.Series([value if i == index else None for i in range(len(df))])
        pbar.update(1)

df.to_csv('data_2.csv')