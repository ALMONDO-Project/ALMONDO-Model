import requests

url = 'https://api.twitter.com/2/usage/tweets'
headers = {
    'Authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAAAB6rAEAAAAAUETTBU7ohCCUuzTnRu1VHgYw4Vk%3DN5I6Yq9m16CwhIjwHsFrYx87qsqBeHxwD1lA6bksneT5IlsIvS'
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    # Process the response data here
    print(data)
else:
    print("Error:", response.status_code)