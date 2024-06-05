# Cleaning text for NLP tasks
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import demoji
from tqdm import tqdm
import sys 

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function to clean tweets
def clean_text(row):
    
    words_to_remove = ['nordisk', 'nordisks', 'novo', 'abb', 'schneider', 'lego', 'hi', 'hello',
    'novorossiysk', 'enel', 'https', 'every', 'single', 'enso', 'acciona', 'akzo', 'today', 'tomorrow',
    'week', 'month', 'whyee', 'june', 'without', 'with', 'last', 'first', 'info', 'good', 'bad', 'little', 'big',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
    'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty',
    'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion', 'trillion',
    'january', 'february', 'march', 'april', 'may', 'july', 'august', 'september', 'october', 'november', 'december',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'other', 'same', 'different', 'new', 'old', 'young', 'long', 'short', 'right', 'wrong', 'high', 'low',
    'early', 'late', 'strong', 'weak', 'happy', 'sad', 'easy', 'difficult', 'hard', 'soft', 'big', 'small',
    'very', 'too', 'also', 'well', 'quickly', 'slowly', 'easily', 'hardly', 'almost', 'nearly', 'always',
    'never', 'sometimes', 'often', 'usually', 'now', 'then', 'here', 'there', 'together', 'apart',
    'and', 'or', 'but', 'nor', 'so', 'yet', 'for', 'xniwrqo', 'wxyk', 'zkdwud', 'dihc', 'jlgt',
    'of', 'to', 'in', 'on', 'at', 'by', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'from', 'up', 'down', 'under', 'over', 'again', 'further',
    'i', 'me', 'you', 'he', 'him', 'she', 'her', 'it', 'we', 'us', 'they', 'them', 'my', 'your', 'his',
    'her', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs', 'abbformulae', 'smnsj',
    'a', 'an', 'the', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'shall', 'ogpj', 'should', 'can', '恭喜发财', 'uaigkgunt', 'could','ycky', 'may', 'might', 'must', 'who', 
    'what', 'when', 'where', 'why', 'how', 'which', 'that', 'this', 'these', 'those', 'there', 'here', 'rjwpjiyti']
    
    def remove_emoji(text):
        for k in demoji.findall(text):
            text = text.replace(k, '')
        return text
    
    def remove_quotes(text):
        text = re.sub(r"[\"\']", ' ', text)
        text = text.replace('"', ' ')
        text = text.replace("'", ' ')
        text = text.replace('“', ' ')
        text = text.replace("’", " ")
        text = text.replace('”', " ")
        return text
    
    text = row['text']
    username = row['username']
       
    # Remove user mentions (words starting with "@")
    text = re.sub(r'@\w+', ' ', text)
    
    # Remove hashtags (words starting with "#")
    text = re.sub(r'#\w+', ' ', text)
    
    # Remove the username of the owner of the tweet
    text = re.sub(rf'\b{username}\b', ' ', text, flags=re.IGNORECASE)
    
    #remove emoji
    text = remove_emoji(text)
       
    # Remove contractions
    contractions_pattern = re.compile(r'\b(?:can\'t|cannot|couldn\'t|aren\'t|they\'re|you\'re|we\'re|I\'m|won\'t|wouldn\'t|shouldn\'t|don\'t|doesn\'t|didn\'t|haven\'t|hasn\'t|hadn\'t|isn\'t|it\'s|wasn\'t|weren\'t|what\'s|that\'s|there\'s|where\'s|who\'s|why\'s|how\'s|i\'ll|you\'ll|he\'ll|she\'ll|we\'ll|they\'ll|i\'d|you\'d|he\'d|she\'d|we\'d|they\'d|let\'s|ain\'t|mustn\'t|needn\'t|shan\'t|can\'ve|could\'ve|may\'ve|might\'ve|must\'ve|shall\'ve|should\'ve|would\'ve|mightn\'t|oughtn\'t|daren\'t|didn\'t|haven\'t|hasn\'t|hadn\'t|isn\'t|wasn\'t|weren\'t|can\'t|couldn\'t|daren\'t|hadn\'t|hasn\'t|haven\'t|isn\'t|mightn\'t|mustn\'t|needn\'t|shan\'t|shouldn\'t|won\'t|wouldn\'t)\b', re.IGNORECASE)
    text = re.sub(contractions_pattern, ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))
    
    # Remove single and double quotes
    text = remove_quotes(text)
    
    # Remove numbers
    text = re.sub(r'\d+', ' ', text)
    
    # Remove newlines, tabs, and extra whitespace
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
        
    # Remove stop words
    stop_words = (stopwords.words('english'))
    #add other useless words
    stop_words.extend(words_to_remove)
    
    #removeing stopwords and words shorter than 4 characters
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    words = [word for word in words if row['username'].lower() not in word] #remove words containing username
    words = [word for word in words if 'http' not in word] #remove strings containing http
    words = [word for word in words if not word.startswith('abb')] #other cleaning steps
    words = [word for word in words if not word.startswith('zmx')] #other cleaning steps
    
    #check after lemmatization
    words = [word for word in words if word not in stop_words]
        
    # Join words back into a single string
    cleaned_text = ' '.join(words)
    
    return cleaned_text


def main():
    df = pd.read_csv(f'data/{sys.argv[1]}', index_col=[0])
    tqdm.pandas()
    df['cleaned_text'] = df.progress_apply(clean_text, axis=1)
    df[['username', 'created_at', 'cleaned_text']].to_csv(f'data/{sys.argv[2]}')


if __name__ == '__main__':
    main()
