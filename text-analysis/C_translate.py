import json
import tqdm
import pandas as pd
from deep_translator import GoogleTranslator

df = pd.read_csv('data/data_3.csv', index_col=[0])

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
for index, row in tqdm.tqdm(df.loc[df.lang != "en"].iterrows(), total=len(df.loc[df.lang != "en"])):
    translated = GoogleTranslator(source="auto", target="en").translate(row.text)
    english_texts[index] = translated

with open('data/translated_texts_2.json', 'w') as file:
    json.dump(english_texts, file)
    
for idx, text in english_texts.items():
    df.loc[idx, "text"] = text

df.to_csv('data/translated_3.csv')