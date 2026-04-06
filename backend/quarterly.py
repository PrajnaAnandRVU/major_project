import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_quarterly_recommendations():
    try:
        # Get S&P 500 data for the last quarter
        spy = yf.Ticker("SPY")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        data = spy.history(start=start_date, end=end_date)
        
        if data.empty:
            return {
                'error': 'Could not fetch market data'
            }
        
        # Calculate market trend
        latest_price = data['Close'].iloc[-1]
        avg_price = data['Close'].mean()
        trend = 'Bullish' if latest_price > avg_price else 'Bearish'
        
        # Get top performing sectors
        sectors = {
            'XLK': 'Technology',
            'XLF': 'Financial',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrial'
        }
        
        sector_performance = []
        for ticker, name in sectors.items():
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                if not hist.empty:
                    performance = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                    sector_performance.append({
                        'name': name,
                        'performance': round(performance, 2)
                    })
            except:
                continue
        
        # Sort sectors by performance
        sector_performance.sort(key=lambda x: x['performance'], reverse=True)
        
        return {
            'market_trend': trend,
            'spy_performance': round(((latest_price - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100, 2),
            'top_sectors': sector_performance[:3],
            'market_outlook': 'Positive' if trend == 'Bullish' else 'Cautious',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        return {
            'error': f'Error generating recommendations: {str(e)}'
        }