
from flask import Blueprint, jsonify
import random

# Reuse sentiment pipeline from news.py
from routes.news import sentiment_pipeline

stock_bp = Blueprint('stock', __name__)

# Dummy stock DB (you can replace with real API later)
STOCKS = {
    "AAPL": {"name": "Apple Inc.", "price": 185},
    "TSLA": {"name": "Tesla Inc.", "price": 250},
    "GOOGL": {"name": "Alphabet Inc.", "price": 140},
    "MSFT": {"name": "Microsoft Corp.", "price": 330},
}

@stock_bp.route('/stock/<symbol>', methods=['GET'])
def get_stock(symbol):
    symbol = symbol.upper()

    if symbol not in STOCKS:
        return jsonify({"error": "Stock not found"}), 404

    stock = STOCKS[symbol]

    # Fake text for sentiment (replace with real news later)
    text = f"{stock['name']} showing strong growth and market expansion"

    analysis = sentiment_pipeline(text)

    return jsonify({
        "symbol": symbol,
        "name": stock["name"],
        "currentPrice": stock["price"],
        "sentiment": analysis["label"],
        "confidence": analysis["confidence"],
        "reasons": analysis["reasons"]
    })
