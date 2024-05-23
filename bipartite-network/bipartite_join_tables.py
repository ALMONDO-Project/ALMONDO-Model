import pandas as pd

#bipartite network dataframe creation
global DATAPATH 
DATAPATH = 'data/in/bipartite_network_tables/'
meeting_participants_lobbyists = pd.read_csv(DATAPATH+'meeting_participants_lobbyists.csv', names = ['id', 'meeting_id', 'lobbyist_id'])
meeting_lobbyists = pd.read_csv(DATAPATH+'meeting_lobbyists.csv', names = ['id', 'identification_code', 'name'])
meetings = pd.read_csv(DATAPATH+'meetings_refactored.csv', on_bad_lines = 'warn', names = ['id', 'cabinet_id', 'dg_id', 'date', 'location', 'subject', 'hash', 'type'])

# Step 1: Merge meeting_participants_lobbyist with meeting_lobbyists
merged_lobbyists = pd.merge(
    meeting_participants_lobbyists, 
    meeting_lobbyists, 
    left_on='lobbyist_id', 
    right_on='id'
)

# Step 2: Merge the result with meetings
final_merged = pd.merge(
    merged_lobbyists, 
    meetings, 
    left_on='meeting_id', 
    right_on='id'
)

# Drop unnecessary columns
final_merged = final_merged.drop(columns=['id_x', 'id_y'])

print('final table has', len(final_merged), 'rows')

final_merged['source'] = final_merged['meeting_id'].apply(lambda x: f'{str(x)}_meeting')
final_merged['target'] = final_merged['lobbyist_id'].apply(lambda x: f'{str(x)}_lobbyist')

final_merged.to_csv('data/out/bipartite_network_tables/bipartite_network_data.csv')