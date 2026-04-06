# from flask import Blueprint, jsonify
# import os
# import requests
# from urllib.parse import quote
# import logging
# import time
# from functools import lru_cache
# from datetime import datetime, timedelta

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
# POLYGON_NEWS_URL = "https://api.polygon.io/v2/reference/news"

# # NewsAPI.org configuration (fallback)
# NEWS_API_KEY = os.getenv("NEWS_API_KEY")
# NEWS_API_URL = "https://newsapi.org/v2/everything"

# # Cache configuration
# NEWS_CACHE_DURATION = 300  # 5 minutes
# NEWS_RATE_LIMIT_DELAY = 1  # 1 second between requests
# last_request_time = 0

# news_bp = Blueprint('news', __name__)

# def validate_api_key():
#     """Validate that the Polygon API key is present and properly formatted"""
#     if not POLYGON_API_KEY:
#         logger.error("Polygon API key is missing")
#         return False
#     if not isinstance(POLYGON_API_KEY, str) or len(POLYGON_API_KEY) < 10:
#         logger.error("Polygon API key appears to be invalid")
#         return False
#     return True

# @lru_cache(maxsize=100)
# def get_cached_news(query: str, timestamp: int) -> list:
#     """Get cached news articles for a query"""
#     return []

# def get_news_from_polygon(query: str, limit: int = 5):
#     """Fetch news from Polygon API"""
#     global last_request_time
    
#     if not validate_api_key():
#         logger.warning("Polygon API key not available")
#         return None

#     try:
#         # Rate limiting
#         time_since_last = time.time() - last_request_time
#         if time_since_last < NEWS_RATE_LIMIT_DELAY:
#             time.sleep(NEWS_RATE_LIMIT_DELAY - time_since_last)

#         # URL encode the query parameter
#         encoded_query = quote(query)
#         params = {
#             "apiKey": POLYGON_API_KEY,
#             "limit": limit,
#             "order": "desc",
#             "sort": "published_utc",
#             "q": encoded_query,
#         }
#         url = POLYGON_NEWS_URL
        
#         logger.info(f"Fetching news from Polygon for query: {query}")
#         response = requests.get(url, params=params, timeout=10)
#         last_request_time = time.time()
        
#         if response.status_code == 401:
#             logger.warning("Polygon authentication failed - check API key")
#             return None
#         elif response.status_code == 429:
#             logger.warning("Polygon rate limit exceeded - will try fallback")
#             return None
#         elif response.status_code != 200:
#             logger.warning(f"Polygon news request failed with status {response.status_code}")
#             return None

#         data = response.json()
#         results = data.get("results", [])[:limit]
#         if not results:
#             logger.warning("Polygon returned no results")
#             return None
            
#         news_list = [
#             {
#                 "title": item.get("title", ""),
#                 "source": (item.get("publisher") or {}).get("name", "Unknown"),
#                 "timestamp": item.get("published_utc", ""),
#                 "content": item.get("description") or "",
#                 "url": item.get("article_url", ""),
#                 "sentiment": 0,
#                 "credibility": 3
#             }
#             for item in results
#         ]
#         logger.info(f"Successfully fetched {len(news_list)} articles from Polygon")
#         return news_list
            
#     except requests.RequestException as e:
#         logger.warning(f"Network error while fetching news from Polygon: {str(e)}")
#         return None
#     except Exception as e:
#         logger.warning(f"Unexpected error fetching news from Polygon: {str(e)}")
#         return None

# def get_news_from_newsapi(query: str, limit: int = 5):
#     """Fetch news from NewsAPI.org as fallback"""
#     if not NEWS_API_KEY:
#         logger.warning("NewsAPI.org key not available")
#         return None
    
#     try:
#         url = f"{NEWS_API_URL}?q={quote(query)}&sortBy=publishedAt&apiKey={NEWS_API_KEY}&language=en"
#         logger.info(f"Fetching news from NewsAPI.org for query: {query}")
#         response = requests.get(url, timeout=10)
        
#         if response.status_code == 401:
#             logger.warning("NewsAPI.org authentication failed - check API key")
#             return None
#         elif response.status_code == 429:
#             logger.warning("NewsAPI.org rate limit exceeded")
#             return None
#         elif response.status_code != 200:
#             logger.warning(f"NewsAPI.org request failed with status {response.status_code}")
#             return None

#         data = response.json()
#         articles = data.get("articles", [])[:limit]
#         if not articles:
#             logger.warning("NewsAPI.org returned no results")
#             return None
            
#         news_list = [
#             {
#                 "title": a.get("title", ""),
#                 "source": (a.get("source") or {}).get("name", "Unknown"),
#                 "timestamp": a.get("publishedAt", ""),
#                 "content": a.get("description") or a.get("content") or "",
#                 "url": a.get("url", ""),
#                 "sentiment": 0,
#                 "credibility": 3
#             }
#             for a in articles if a.get("title")  # Filter out articles without titles
#         ]
#         logger.info(f"Successfully fetched {len(news_list)} articles from NewsAPI.org")
#         return news_list
        
#     except requests.RequestException as e:
#         logger.warning(f"Network error while fetching news from NewsAPI.org: {str(e)}")
#         return None
#     except Exception as e:
#         logger.warning(f"Unexpected error fetching news from NewsAPI.org: {str(e)}")
#         return None

# def get_news(query: str, limit: int = 5):
#     """Fetch news articles with automatic fallback between Polygon and NewsAPI.org"""
#     # Check cache first
#     current_time = int(time.time())
#     cache_key = f"{query}_{current_time // NEWS_CACHE_DURATION}"
#     cached_news = get_cached_news(cache_key, current_time)
#     if cached_news:
#         logger.info(f"Returning cached news for query: {query}")
#         return cached_news

#     # Try Polygon first
#     news_list = get_news_from_polygon(query, limit)
    
#     # Fallback to NewsAPI.org if Polygon fails
#     if not news_list:
#         logger.info(f"Polygon failed, trying NewsAPI.org as fallback for query: {query}")
#         news_list = get_news_from_newsapi(query, limit)
    
#     # If both fail, return demo news
#     if not news_list:
#         logger.warning(f"Both Polygon and NewsAPI.org failed for query: {query}, returning demo news")
#         return get_demo_news()
    
#     # Cache successful results
#     get_cached_news.cache_clear()
#     get_cached_news(cache_key, current_time)
#     return news_list

# def get_demo_news():
#     """Return demo news articles when API fails"""
#     return [
#         {
#             "title": "Demo Market Rally",
#             "source": "DemoSource",
#             "timestamp": datetime.now().isoformat(),
#             "content": "Stocks rallied today as investors cheered strong earnings.",
#             "url": "https://example.com",
#             "sentiment": 0.5,
#             "credibility": 4
#         },
#         {
#             "title": "Demo Fed Rate Decision",
#             "source": "DemoSource",
#             "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
#             "content": "The Federal Reserve held rates steady, citing stable inflation.",
#             "url": "https://example.com",
#             "sentiment": 0.1,
#             "credibility": 3
#         },
#         {
#             "title": "Demo Tech Stocks Surge",
#             "source": "DemoSource",
#             "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
#             "content": "Tech stocks led the market higher on new AI breakthroughs.",
#             "url": "https://example.com",
#             "sentiment": 0.7,
#             "credibility": 5
#         }
#     ]

# @news_bp.route('/news', methods=['GET'])
# def news():
#     """Get latest general financial news (not filtered by ticker)"""
#     try:
#         news_list = get_news('finance', limit=10)
#         return jsonify(news_list)
#     except Exception as e:
#         logger.error(f"Error in news endpoint: {str(e)}")
#         return jsonify(get_demo_news()) 

from flask import Blueprint, jsonify
import os
import requests
from urllib.parse import quote
import logging
import time
from functools import lru_cache
from datetime import datetime, timedelta

# 🔥 NEW: Sentiment + Explainability imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from keybert import KeyBERT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
POLYGON_NEWS_URL = "https://api.polygon.io/v2/reference/news"

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

# Cache config
NEWS_CACHE_DURATION = 300
NEWS_RATE_LIMIT_DELAY = 1
last_request_time = 0

news_bp = Blueprint('news', __name__)

# 🔥 Load models once (important)
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
kw_model = KeyBERT()

# ---------------- SENTIMENT FUNCTIONS ---------------- #

def analyze_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        score = probs.detach().numpy()[0]

        labels = ["negative", "neutral", "positive"]
        sentiment = labels[score.argmax()]
        confidence = float(score.max())

        return sentiment, confidence
    except Exception as e:
        logger.warning(f"Sentiment error: {str(e)}")
        return "neutral", 0.5


def extract_keywords(text):
    try:
        keywords = kw_model.extract_keywords(text, top_n=3)
        return [kw[0] for kw in keywords]
    except Exception as e:
        logger.warning(f"Keyword extraction error: {str(e)}")
        return ["market movement", "financial news"]


def sentiment_pipeline(text):
    sentiment, confidence = analyze_sentiment(text)
    keywords = extract_keywords(text)

    label = "Bullish" if sentiment == "positive" else "Bearish"

    return {
        "sentiment": confidence if sentiment == "positive" else -confidence,
        "label": label,
        "confidence": round(confidence * 100, 2),
        "reasons": keywords
    }

# ---------------- HELPERS ---------------- #

def validate_api_key():
    if not POLYGON_API_KEY:
        logger.error("Polygon API key is missing")
        return False
    if not isinstance(POLYGON_API_KEY, str) or len(POLYGON_API_KEY) < 10:
        logger.error("Polygon API key appears to be invalid")
        return False
    return True


@lru_cache(maxsize=100)
def get_cached_news(query: str, timestamp: int) -> list:
    return []


# ---------------- POLYGON ---------------- #

def get_news_from_polygon(query: str, limit: int = 5):
    global last_request_time

    if not validate_api_key():
        return None

    try:
        time_since_last = time.time() - last_request_time
        if time_since_last < NEWS_RATE_LIMIT_DELAY:
            time.sleep(NEWS_RATE_LIMIT_DELAY - time_since_last)

        encoded_query = quote(query)
        params = {
            "apiKey": POLYGON_API_KEY,
            "limit": limit,
            "order": "desc",
            "sort": "published_utc",
            "q": encoded_query,
        }

        response = requests.get(POLYGON_NEWS_URL, params=params, timeout=10)
        last_request_time = time.time()

        if response.status_code != 200:
            return None

        data = response.json()
        results = data.get("results", [])[:limit]

        news_list = []

        for item in results:
            title = item.get("title", "")
            content = item.get("description") or ""

            analysis = sentiment_pipeline(title + " " + content)

            news_list.append({
                "title": title,
                "source": (item.get("publisher") or {}).get("name", "Unknown"),
                "timestamp": item.get("published_utc", ""),
                "content": content,
                "url": item.get("article_url", ""),
                "sentiment": analysis["sentiment"],
                "label": analysis["label"],
                "confidence": analysis["confidence"],
                "reasons": analysis["reasons"],
                "credibility": 3
            })

        return news_list if news_list else None

    except Exception as e:
        logger.warning(f"Polygon error: {str(e)}")
        return None


# ---------------- NEWSAPI ---------------- #

def get_news_from_newsapi(query: str, limit: int = 5):
    if not NEWS_API_KEY:
        return None

    try:
        url = f"{NEWS_API_URL}?q={quote(query)}&sortBy=publishedAt&apiKey={NEWS_API_KEY}&language=en"
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()
        articles = data.get("articles", [])[:limit]

        news_list = []

        for a in articles:
            title = a.get("title", "")
            content = a.get("description") or a.get("content") or ""

            if not title:
                continue

            analysis = sentiment_pipeline(title + " " + content)

            news_list.append({
                "title": title,
                "source": (a.get("source") or {}).get("name", "Unknown"),
                "timestamp": a.get("publishedAt", ""),
                "content": content,
                "url": a.get("url", ""),
                "sentiment": analysis["sentiment"],
                "label": analysis["label"],
                "confidence": analysis["confidence"],
                "reasons": analysis["reasons"],
                "credibility": 3
            })

        return news_list if news_list else None

    except Exception as e:
        logger.warning(f"NewsAPI error: {str(e)}")
        return None


# ---------------- MAIN LOGIC ---------------- #

def get_news(query: str, limit: int = 5):
    current_time = int(time.time())
    cache_key = f"{query}_{current_time // NEWS_CACHE_DURATION}"

    cached_news = get_cached_news(cache_key, current_time)
    if cached_news:
        return cached_news

    news_list = get_news_from_polygon(query, limit)

    if not news_list:
        news_list = get_news_from_newsapi(query, limit)

    if not news_list:
        return get_demo_news()

    get_cached_news.cache_clear()
    get_cached_news(cache_key, current_time)

    return news_list


# ---------------- DEMO ---------------- #

def get_demo_news():
    return [
        {
            "title": "Market Rally on Strong Earnings",
            "source": "DemoSource",
            "timestamp": datetime.now().isoformat(),
            "content": "Stocks rallied today due to strong earnings reports.",
            "url": "https://example.com",
            "sentiment": 0.6,
            "label": "Bullish",
            "confidence": 78,
            "reasons": ["strong earnings", "market rally"],
            "credibility": 4
        },
        {
            "title": "Tech Stocks Fall Amid Uncertainty",
            "source": "DemoSource",
            "timestamp": datetime.now().isoformat(),
            "content": "Tech stocks declined due to market uncertainty.",
            "url": "https://example.com",
            "sentiment": -0.5,
            "label": "Bearish",
            "confidence": 70,
            "reasons": ["market uncertainty", "stock decline"],
            "credibility": 3
        }
    ]


# ---------------- ROUTE ---------------- #

@news_bp.route('/news', methods=['GET'])
def news():
    try:
        news_list = get_news('finance', limit=10)
        return jsonify(news_list)
    except Exception as e:
        logger.error(f"Error in news endpoint: {str(e)}")
        return jsonify(get_demo_news())