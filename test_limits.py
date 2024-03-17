import tweepy

BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAAB6rAEAAAAAUETTBU7ohCCUuzTnRu1VHgYw4Vk%3DN5I6Yq9m16CwhIjwHsFrYx87qsqBeHxwD1lA6bksneT5IlsIvS'


client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

print('>>> request with max_results = 1000 and limit = 5')
for response in tweepy.Paginator(client.get_users_tweets, 606342802, max_results=1000, limit=5): #questo dovrebbe voler dire massimo 1000 tweet in 5 richieste
    print(response.meta)

print('>>> request with max_results = 10 and limit = 5')    
for response in tweepy.Paginator(client.get_users_tweets, 606342802, max_results=10, limit=5): #massimo 10 tweet in 5 richieste
    print(response.meta)

print('>>> request with max_results = 1000 and no limit')        
for response in tweepy.Paginator(client.get_users_tweets, 606342802, max_results=100): #massimo 100 tweet in qualunque numero di richieste
    print(response.meta)
    
#ma non capisco se è come penso io oppure no e ho paura che se lancio codice poi mi consuma tutti i tweet