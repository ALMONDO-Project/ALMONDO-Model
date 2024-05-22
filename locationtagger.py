import pandas as pd
import locationtagger
import re
from tqdm import tqdm
#location recognition
global DATAPATH 
DATAPATH = 'data/in/bipartite_network_tables/'
FILENAME = 'bipartite_network_data.csv'
final_merged = pd.read_csv(DATAPATH+FILENAME, index_col = [0])
def locationtag(x):
    entities = locationtagger.find_locations(text = re.sub(r'[^\w\s]', '', x.lower().strip()))
    pbar.update(1)
    try:
        return entities.cities[0]
    except:
        return x
        
with tqdm(total=len(final_merged)) as pbar:    
    final_merged['location'] = final_merged['location'].apply(lambda x: locationtag(x))

final_merged.to_csv(DATAPATH+FILENAME)