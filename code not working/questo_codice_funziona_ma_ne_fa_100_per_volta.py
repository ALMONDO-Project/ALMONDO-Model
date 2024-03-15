from datetime import datetime
import requests

BEARER_TOKEN=''

def get_user_tweets(user_id, start_time, end_time):
    tweet_count = 0
    tweets = []
    max_results_per_request = 100  # Maximum results per request
    max_requests_per_15mins = 10   # Maximum requests per 15 minutes

    # Convert start_time and end_time to Twitter API format
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Pagination token
    next_token = None

    while True:
        try:
            # Check rate limits
            # This is not supported directly with Bearer token authentication
            # Make requests without rate limiting

            # Construct URL
            url = f"https://api.twitter.com/2/users/{user_id}/tweets?start_time={start_time_str}&end_time={end_time_str}&max_results={max_results_per_request}"
            if next_token:
                url += f"&next_token={next_token}"

            # Make request
            headers = {
                "Authorization": f"Bearer {BEARER_TOKEN}",
                "Content-Type": "application/json"
            }
            response = requests.get(url, headers=headers)
            response_data = response.json()

            # Check if response contains data
            if 'data' in response_data:
                # Add tweets to the list
                tweets.extend(response_data['data'])

            # Update pagination token
            if 'meta' in response_data and 'next_token' in response_data['meta']:
                next_token = response_data['meta']['next_token']
            else:
                break  # No more tweets available

            # Check if reached the end of the time range or tweet limit
            if tweet_count >= 3200:
                break

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break

    return tweets

# Example usage
user_id = '156646851'
start_time = datetime(20234, 1, 1)
end_time = datetime(2024, 1, 31)

user_tweets = get_user_tweets(user_id, start_time, end_time)
print(f"Retrieved {len(user_tweets)} tweets.")