import unittest
from unittest.mock import MagicMock, patch
from downloadData import *

class TestUserDataDownloader(unittest.TestCase):
    def setUp(self):
        # Initialize UserDataDownloader with mock parameters
        self.downloader = UserDataDownloader(username='test_user')

    def tearDown(self):
        pass

    def test_make_dirs(self):
        # Test directory creation
        with patch('os.makedirs') as mock_makedirs:
            self.downloader.make_dirs()
            mock_makedirs.assert_called_once_with(self.downloader.tweets_path)

    def test_set_max_id(self):
        # Test setting max_id
        with patch('your_module.get_oldest_tweet_id') as mock_get_oldest_tweet_id:
            mock_get_oldest_tweet_id.return_value = '123456789'
            self.downloader.set_max_id()
            self.assertEqual(self.downloader.max_id, '123456789')

    def test_set_max_tweets(self):
        # Test setting max_tweets
        with patch('your_module.compute_max_tweets') as mock_compute_max_tweets:
            mock_compute_max_tweets.return_value = 1100  # Mocking remaining API quota
            self.downloader.set_max_tweets(BEARER_TOKEN='mock_token', n=None)
            self.assertEqual(self.downloader.max_tweets, 1000)

            self.downloader.set_max_tweets(BEARER_TOKEN='mock_token', n=500)
            self.assertEqual(self.downloader.max_tweets, 500)

        # Test NoTweetsLeftException
        with patch('your_module.compute_max_tweets') as mock_compute_max_tweets:
            mock_compute_max_tweets.return_value = 900  # Mocking remaining API quota
            with self.assertRaises(NoTweetsLeftException):
                self.downloader.set_max_tweets(BEARER_TOKEN='mock_token')

    def test_download(self):
        # Mock collect_results function
        with patch('your_module.collect_results') as mock_collect_results:
            mock_collect_results.return_value = [{'id': '123'}, {'id': '456'}]
            self.downloader.download()
            self.assertEqual(len(self.downloader.tweets), 2)

    def test_save_tweets(self):
        # Test saving tweets
        self.downloader.tweets = [{'id': '123', 'text': 'Test tweet 1'}, {'id': '456', 'text': 'Test tweet 2'}]
        with patch('json.dump') as mock_json_dump:
            self.downloader.save_tweets()
            mock_json_dump.assert_called_once()

        # Test NoTweetsToSaveException
        self.downloader.tweets = []
        with self.assertRaises(NoTweetsToSaveException):
            self.downloader.save_tweets()

if __name__ == '__main__':
    unittest.main()
