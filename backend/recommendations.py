from flask import Blueprint, jsonify
import os
import requests
from news import get_news
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from groq import Groq

recommendations_bp = Blueprint('recommendations', __name__)

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
POLYGON_BASE_URL = "https://api.polygon.io"

# For demo, use a static list of tickers
TOP_TICKERS = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL"]

# Simple in-memory cache
_recommendations_cache = {"data": None, "timestamp": 0}
CACHE_TTL = 30  # seconds - reduced from 60 to ensure fresher data

def get_polygon_data(ticker):
    try:
        # Get real-time price
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{ticker}/prev?adjusted=true&apiKey={POLYGON_API_KEY}"
        resp = requests.get(url)
        price = None
        if resp.status_code == 200 and resp.json().get("results"):
            price = resp.json()["results"][0]["c"]
        # Get company name
        url2 = f"{POLYGON_BASE_URL}/v3/reference/tickers/{ticker}?apiKey={POLYGON_API_KEY}"
        resp2 = requests.get(url2)
        name = ticker
        if resp2.status_code == 200 and resp2.json().get("results"):
            name = resp2.json()["results"].get("name", ticker)
        return {"ticker": ticker, "company_name": name, "price": price}
    except Exception as e:
        print(f"Polygon error for {ticker}: {e}")
        return {"ticker": ticker, "company_name": ticker, "price": None}

def get_groq_analysis(prompt):
    try:
        client = Groq()
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "system", "content": "You are a financial research assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        return "No AI analysis available."

def prompt_groq_for_recommendations(stocks_data):
    prompt = (
        "You are a world-class financial AI. Given the following real-time stock data and news, "
        "analyze and recommend the top 3 stocks to buy right now. For each, provide:\n"
        "- Ticker\n- Company Name\n- Sentiment Score (-1 to 1)\n- Short summary (1-2 sentences)\n"
        "Respond in JSON as an array of objects with keys: ticker, company_name, sentiment, summary.\n"
        "Here is the data:\n"
    )
    for stock in stocks_data:
        prompt += (
            f"\nTicker: {stock['ticker']}\n"
            f"Company: {stock['company_name']}\n"
            f"Price: {stock['price']}\n"
            f"News:\n"
        )
        for article in stock.get("news", [])[:3]:
            prompt += f"- {article.get('title','')} ({article.get('source','')}): {article.get('content','')}\n"
    # Call Groq API
    recs_str = get_groq_analysis(prompt)
    # Log the raw response for debugging
    print("Raw Groq response:\n", recs_str)
    # Clean the response
    cleaned = ''.join(ch for ch in recs_str if ord(ch) >= 32 or ch in '\n\t')
    # Extract JSON array
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except Exception as e:
            print(f"JSON parse error: {e}")
            print("Raw JSON string:", json_str)
            return []
    else:
        print("No JSON array found in Groq response.")
        print("Cleaned response:\n", cleaned)
        return []

def get_recommendations_cached():
    now = time.time()
    # Clear cache if it's expired
    if _recommendations_cache["data"] and now - _recommendations_cache["timestamp"] < CACHE_TTL:
        logging.info(f"Returning cached recommendations (age: {now - _recommendations_cache['timestamp']:.1f}s)")
        return _recommendations_cache["data"]
    
    # Cache expired or doesn't exist, recompute
    logging.info("Cache expired or empty, computing new recommendations")
    try:
        data = compute_recommendations()
        # Ensure we always return at least 3 recommendations
        if not data or len(data) < 3:
            # Fill remaining slots with neutral recommendations
            existing_tickers = {d["ticker"] for d in data}
            for ticker in TOP_TICKERS:
                if len(data) >= 3:
                    break
                if ticker not in existing_tickers:
                    data.append({
                        "ticker": ticker,
                        "company_name": ticker,
                        "sentiment": 0.0,
                        "summary": "Limited news data available for analysis."
                    })
        _recommendations_cache["data"] = data
        _recommendations_cache["timestamp"] = now
        return data[:3]  # Always return exactly 3
    except Exception as e:
        logging.error(f"Error computing recommendations: {e}")
        # Fallback: return neutral recommendations
        return [{"ticker": t, "company_name": t, "sentiment": 0.0, "summary": "Analysis temporarily unavailable."} for t in TOP_TICKERS[:3]]

def calculate_sentiment_from_news(news_articles):
    """Calculate sentiment score from news articles using keyword analysis"""
    if not news_articles or len(news_articles) == 0:
        return 0.0
    
    positive_words = ['growth', 'beat', 'surge', 'rally', 'gain', 'profit', 'strong', 'up', 'bullish', 'positive', 
                     'win', 'success', 'rise', 'increase', 'soar', 'climb', 'advance', 'outperform', 'exceed', 
                     'exceeded', 'exceeds', 'record', 'high', 'peak', 'optimistic', 'favorable', 'boosts', 
                     'improves', 'improved', 'stronger', 'momentum', 'breakthrough', 'expansion', 'acquisition']
    negative_words = ['fall', 'drop', 'decline', 'loss', 'weak', 'down', 'bearish', 'negative', 'fail', 'miss', 
                     'worry', 'concern', 'plunge', 'crash', 'tumble', 'sink', 'dive', 'decrease', 'underperform', 
                     'disappoint', 'disappointed', 'disappoints', 'low', 'bottom', 'pessimistic', 'unfavorable', 
                     'drops', 'deteriorates', 'deteriorated', 'weaker', 'slowdown', 'recession', 'crisis', 
                     'layoff', 'layoffs', 'cut', 'cuts', 'reduction']
    
    sentiment_scores = []
    for article in news_articles:
        title = article.get('title', '') or ''
        content = article.get('content', '') or article.get('summary', '') or ''
        title_content = (title + ' ' + content).lower()
        
        # Count keywords
        title_text = title.lower()
        content_text = content.lower()
        
        title_pos = sum(1 for word in positive_words if word in title_text)
        title_neg = sum(1 for word in negative_words if word in title_text)
        content_pos = sum(1 for word in positive_words if word in content_text)
        content_neg = sum(1 for word in negative_words if word in content_text)
        
        # Weight title more heavily (2x)
        weighted_pos = (title_pos * 2) + content_pos
        weighted_neg = (title_neg * 2) + content_neg
        
        # Calculate sentiment
        if weighted_pos + weighted_neg > 0:
            sentiment_score = (weighted_pos - weighted_neg) / max(weighted_pos + weighted_neg, 1)
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
        else:
            # No keywords - neutral
            sentiment_score = 0.0
        
        sentiment_scores.append(sentiment_score)
    
    # Return average sentiment
    if sentiment_scores:
        return round(sum(sentiment_scores) / len(sentiment_scores), 2)
    return 0.0

def compute_recommendations():
    with ThreadPoolExecutor() as executor:
        stock_futures = {executor.submit(get_polygon_data, ticker): ticker for ticker in TOP_TICKERS}
        news_futures = {executor.submit(get_news, ticker, 5): ticker for ticker in TOP_TICKERS}
        stocks_data = []
        news_data = {}
        for future in as_completed(stock_futures):
            ticker = stock_futures[future]
            try:
                stocks_data.append(future.result())
            except Exception:
                stocks_data.append({"ticker": ticker, "company_name": ticker, "price": None})
        for future in as_completed(news_futures):
            ticker = news_futures[future]
            try:
                news_data[ticker] = future.result()
            except Exception:
                news_data[ticker] = []
        for stock in stocks_data:
            stock["news"] = news_data.get(stock["ticker"], [])
    
    # Calculate sentiment for each stock from news
    recommendations = []
    for stock in stocks_data:
        ticker = stock["ticker"]
        news_articles = stock.get("news", [])
        sentiment = calculate_sentiment_from_news(news_articles)
        
        logging.info(f"Calculated sentiment for {ticker}: {sentiment} (from {len(news_articles)} articles)")
        
        # Generate summary based on sentiment
        if sentiment > 0.3:
            summary = f"Strong positive sentiment detected. Recent news suggests favorable outlook for {ticker}."
        elif sentiment > 0:
            summary = f"Moderate positive sentiment. {ticker} shows promising indicators from recent coverage."
        elif sentiment < -0.2:
            summary = f"Negative sentiment detected. Recent news suggests concerns around {ticker}."
        else:
            summary = f"Neutral sentiment. {ticker} shows mixed signals from recent news coverage."
        
        recommendations.append({
            "ticker": ticker,
            "company_name": stock.get("company_name", ticker),
            "sentiment": sentiment,
            "summary": summary
        })
    
    # Sort by sentiment (highest first) and return top 3
    recommendations.sort(key=lambda x: x["sentiment"], reverse=True)
    logging.info(f"Sorted recommendations: {[(r['ticker'], r['sentiment']) for r in recommendations]}")
    return recommendations[:3]

@recommendations_bp.route('/recommendations', methods=['GET'])
def recommendations():
    try:
        recs = get_recommendations_cached()
        logging.info(f"Returning {len(recs)} recommendations: {[(r['ticker'], r['sentiment']) for r in recs]}")
        print(f"\n{'='*60}")
        print(f"RECOMMENDATIONS ENDPOINT")
        print(f"Returning {len(recs)} recommendations:")
        for r in recs:
            print(f"  - {r['ticker']}: sentiment={r['sentiment']}, summary={r['summary'][:50]}")
        print(f"{'='*60}\n")
        
        # Ensure response has proper headers
        response = jsonify(recs)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
        return response
    except Exception as e:
        logging.error(f"Error in recommendations endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500 