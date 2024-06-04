import json
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from deep_translator import GoogleTranslator
from langdetect import detect
from tqdm.notebook import tqdm

def format_username(username):
    # Dictionary for specific replacements
    specific_replacements = {
        'lego_group': 'Lego Group',
        'hm': 'H&M',
        'detushepostdhl': 'DHL',
        'iberdrola_en': 'Iberdrola'
    }
    # Apply specific replacements if available
    if username in specific_replacements:
        return specific_replacements[username]
    # General case for handling 'group'
    if 'group' in username.lower():
        parts = username.split('group')
        return ' '.join([part.capitalize() for part in parts if part]) + ' Group'
    # Capitalize other usernames
    return username.capitalize()

df = pd.read_csv('data_2.csv', index_col=[0])
df = df[['username', 'lang', 'text', 'created_at']]

#remove qme #media links 
df = df.loc[df.lang != 'qme']
#remove zxx #used when the language is unknown
df = df.loc[df.lang != 'zxx']
#If no language classification can be made for a Tweet
df = df.loc[df.lang != 'und']
#tweets with hashtags only
df = df.loc[df.lang != 'qht']
#lang:qam for tweets with mentions only (works for tweets since 2022-06
df = df.loc[df.lang != 'qam']
#questo non so che è ma lo toglierei
df = df.loc[df.lang != 'art']

no_english = df.loc[df.lang != 'en']

print(f"We have {len(df.loc[df.lang != 'en'])} tweets in languages that are not English.")

print(f"We have {len(df.loc[df.lang != 'en'].username.unique())} different companies tweeting in non-english languages:")

print(f"we have {len(df.lang.unique())} different languages")

print("such languages are:")

df.loc[df.lang != 'en'].lang.unique()

english_texts = {}
for index, row in tqdm(df.loc[df.new_lang != "en"].iterrows(), total=len(df)):
    translated = GoogleTranslator(source="auto", target="en").translate(row.text)
    english_texts[index] = translated
    
for idx, text in english_texts.items():
    df.loc[idx, "text"] = text
df.drop('en_text', axis=1)
for index, row in no_english.iterrows():
    if index in df.index:
        df.loc[index, 'text'] = row['en_text']
df = df.drop(columns=['en_text'])

df[['username', 'text', 'created_at', 'new_lang']]\
    .to_csv('translated_2.csv')