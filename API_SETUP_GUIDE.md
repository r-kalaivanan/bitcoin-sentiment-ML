# 🚀 API Setup Guide for Bitcoin Sentiment ML

## 📋 Overview

This guide will help you set up real API keys for Twitter and Reddit to get live sentiment data instead of mock data.

## 🐦 Twitter API Setup (Already Configured!)

✅ **Your Twitter API is already configured!**

- Bearer Token: Already in your .env file
- You should be getting real Twitter data

## 📱 Reddit API Setup

### Step 1: Create a Reddit App

1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Fill in the form:
   - **Name**: Bitcoin Sentiment Analyzer
   - **App type**: Select "script"
   - **Description**: ML sentiment analysis for Bitcoin
   - **About URL**: Leave empty
   - **Redirect URI**: http://localhost:8080

### Step 2: Get Your Credentials

After creating the app, you'll see:

- **Client ID**: This is the string under "personal use script" (14 characters)
- **Client Secret**: This is the "secret" field (27 characters)

### Step 3: Update Your .env File

Replace these lines in your `.env` file:

```bash
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
```

With your actual credentials:

```bash
REDDIT_CLIENT_ID=your_actual_14_char_id
REDDIT_CLIENT_SECRET=your_actual_27_char_secret
```

## 📰 News API Setup (Optional)

### Free Option - NewsAPI

1. Go to https://newsapi.org/register
2. Sign up for a free account (100 requests/day)
3. Copy your API key
4. Update your .env file:

```bash
NEWS_API_KEY=your_actual_news_api_key
```

## 🔧 Testing Your Setup

After updating your API keys, run this test:

```bash
python scripts/api_test.py
```

This will test all your API connections and show you if they're working properly.

## 🚨 Important Notes

1. **Keep your .env file secure** - Never commit it to GitHub
2. **API Rate Limits**:

   - Twitter: 300 requests per 15 minutes
   - Reddit: 60 requests per minute
   - NewsAPI: 100 requests per day (free tier)

3. **If you don't set up APIs**: The system will use mock data (which still works for testing)

## 🎯 Expected Results

With real APIs configured:

- **Twitter**: 50-100 real tweets about Bitcoin
- **Reddit**: 30-50 real posts from r/bitcoin and related subreddits
- **News**: 10-20 recent news articles about Bitcoin

This will give you much more accurate and current sentiment analysis!
