# scripts/scrape.py

import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import snscrape.modules.twitter as sntwitter
import pandas as pd

def scrape_tweets(query, since, until, max_results=100):
    tweets = []
    scraper = sntwitter.TwitterSearchScraper(f'{query} since:{since} until:{until}')

    for i, tweet in enumerate(scraper.get_items()):
        if i >= max_results:
            break
        tweets.append({
            'date': tweet.date,
            'username': tweet.user.username,
            'content': tweet.content
        })

    return pd.DataFrame(tweets)

if __name__ == "__main__":
    df = scrape_tweets("bitcoin OR btc", "2023-01-01", "2023-01-02", max_results=200)
    print(df.head())
    df.to_csv("data/raw_tweets.csv", index=False)
