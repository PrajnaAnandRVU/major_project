# import yfinance as yf
# from transformers import pipeline

# # Load model once (IMPORTANT)
# sentiment_analyzer = pipeline("sentiment-analysis")

# def get_sentiment(ticker):
#     try:
#         stock = yf.Ticker(ticker)
#         news = stock.news

#         if not news:
#             return {'error': 'No news articles found for this ticker'}

#         sentiments = []

#         for article in news[:10]:  # Keep it reasonable
#             title = article.get('title', '')
#             summary = article.get('summary', '')
#             text = f"{title}. {summary}"

#             if not text.strip():
#                 continue

#             result = sentiment_analyzer(text[:512])[0]  # truncate long text

#             sentiments.append({
#                 'title': title,
#                 'sentiment': result['label'],
#                 'score': round(result['score'], 3)
#             })

#         if not sentiments:
#             return {'error': 'Could not analyze sentiment'}

#         # Better scoring system
#         total = 0
#         for s in sentiments:
#             if s['sentiment'] == 'POSITIVE':
#                 total += s['score']
#             elif s['sentiment'] == 'NEGATIVE':
#                 total -= s['score']
#             else:
#                 total += 0  # neutral

#         sentiment_score = round(total / len(sentiments), 3)

#         # Interpretation
#         if sentiment_score > 0.3:
#             recommendation = "Strong positive sentiment → Buy/Hold"
#         elif sentiment_score > 0:
#             recommendation = "Mild positive sentiment → Cautious Hold"
#         elif sentiment_score > -0.3:
#             recommendation = "Mild negative sentiment → Be Careful"
#         else:
#             recommendation = "Strong negative sentiment → Sell/Reduce"

#         return {
#             'sentiment_score': sentiment_score,
#             'articles_analyzed': len(sentiments),
#             'key_findings': sentiments,
#             'recommendation': recommendation
#         }

#     except Exception as e:
#         return {'error': str(e)}


import os
import math
from datetime import datetime
import yfinance as yf
from newsapi import NewsApiClient
from transformers import pipeline

# ==============================
# 🔐 LOAD API KEY
# ==============================
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ==============================
# 🧠 LOAD FINBERT SAFELY
# ==============================
def load_finbert():
    try:
        analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            framework="pt",
            device=-1  # CPU safe
        )

        print("✅ FinBERT loaded successfully")
        print("MODEL:", analyzer.model.name_or_path)

        return analyzer

    except Exception as e:
        print("❌ FinBERT failed, fallback model loaded:", e)

        fallback = pipeline("sentiment-analysis")

        print("Fallback MODEL:", fallback.model.name_or_path)

        return fallback


# Load once globally
sentiment_analyzer = load_finbert()

# ==============================
# 📰 NEWS FETCHING
# ==============================
newsapi = NewsApiClient(api_key=NEWS_API_KEY)


def fetch_news(ticker):
    articles = []

    # 🔹 NewsAPI (primary)
    try:
        news = newsapi.get_everything(
            q=ticker,
            language="en",
            sort_by="publishedAt",
            page_size=10
        )
        articles.extend(news.get("articles", []))
    except Exception as e:
        print("NewsAPI failed:", e)

    # 🔹 Yahoo Finance (fallback)
    try:
        stock = yf.Ticker(ticker)
        yf_news = stock.news

        for article in yf_news[:5]:
            articles.append({
                "title": article.get("title"),
                "description": article.get("summary"),
                "publishedAt": article.get("providerPublishTime")
            })
    except Exception as e:
        print("Yahoo Finance failed:", e)

    return articles


# ==============================
# 📊 SENTIMENT FUNCTION
# ==============================
def get_sentiment(ticker):
    try:
        print(f"\n🔍 Running sentiment for: {ticker}")

        articles = fetch_news(ticker)

        if not articles:
            return {"error": "No news articles found"}

        sentiments = []
        total_score = 0
        total_weight = 0

        now = datetime.utcnow()

        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            content = f"{title}. {description}"

            if not content.strip():
                continue

            # 🧠 Run FinBERT
            result = sentiment_analyzer(content[:512])[0]

            label = result['label'].lower()
            confidence = result['score']

            # 🔢 Convert label to numeric
            if label == "positive":
                numeric_score = confidence
            elif label == "negative":
                numeric_score = -confidence
            else:
                numeric_score = 0

            # ⏳ Recency weighting
            published_at = article.get("publishedAt")

            try:
                if isinstance(published_at, int):  # Yahoo timestamp
                    article_time = datetime.utcfromtimestamp(published_at)
                else:
                    article_time = datetime.strptime(
                        published_at, "%Y-%m-%dT%H:%M:%SZ"
                    )

                hours_old = (now - article_time).total_seconds() / 3600
                weight = math.exp(-hours_old / 24)

            except:
                weight = 1  # fallback

            weighted_score = numeric_score * weight

            total_score += weighted_score
            total_weight += weight

            sentiments.append({
                "title": title,
                "sentiment": label.upper(),
                "confidence": round(confidence, 3),
                "weight": round(weight, 3),
                "impact_score": round(weighted_score, 3)
            })

        if total_weight == 0:
            return {"error": "No valid sentiment data"}

        # 📊 Final weighted score
        final_score = round(total_score / total_weight, 3)

        # 📈 Label decision
        if final_score > 0.2:
            final_label = "BULLISH"
        elif final_score < -0.2:
            final_label = "BEARISH"
        else:
            final_label = "NEUTRAL"

        # 🧪 DEBUG CHECK (IMPORTANT)
        print("MODEL USED:", sentiment_analyzer.model.name_or_path)
        print("TEST SAMPLE:", sentiment_analyzer("Strong earnings growth"))

        # 🧠 Explanation
        explanation = (
            f"The stock shows a {final_label.lower()} sentiment with a weighted score of {final_score}. "
            f"This uses FinBERT (finance-specific NLP model) across {len(sentiments)} articles. "
            f"Recent news has higher influence due to time-decay weighting, meaning newer developments impact the score more."
        )

        return {
            "ticker": ticker,
            "final_score": final_score,
            "label": final_label,
            "articles_analyzed": len(sentiments),
            "articles": sentiments,
            "explanation": explanation
        }

    except Exception as e:
        return {"error": str(e)}