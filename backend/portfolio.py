import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns

def optimize_portfolio(holdings_str):
    try:
        # Parse holdings string (format: "AAPL:100,MSFT:50")
        holdings_list = [h.strip() for h in holdings_str.split(',')]
        holdings = {}
        
        for holding in holdings_list:
            if ':' not in holding:
                continue
            parts = holding.split(':')
            if len(parts) != 2:
                continue
            ticker, amount = parts
            try:
                holdings[ticker.strip()] = float(amount.strip())
            except ValueError:
                continue
        
        if not holdings:
            return {
                'error': 'Invalid portfolio format. Please use format: AAPL:100,MSFT:50',
                'allocations': [],
                'performance': None
            }
        
        # Get historical data
        tickers = list(holdings.keys())
        data = pd.DataFrame()
        
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')
            if not hist.empty:
                data[ticker] = hist['Close']
        
        if data.empty:
            return {
                'error': 'Could not fetch stock data',
                'allocations': [],
                'performance': None
            }
        
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        
        # Optimize portfolio
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
        # Calculate performance metrics
        performance = ef.portfolio_performance()
        
        # Format results
        allocations = []
        for ticker, weight in cleaned_weights.items():
            if weight > 0:
                allocations.append({
                    'ticker': ticker,
                    'current_amount': holdings[ticker],
                    'optimal_weight': round(weight * 100, 2),
                    'optimal_amount': round(holdings[ticker] * weight / cleaned_weights[ticker], 2)
                })
        
        return {
            'allocations': allocations,
            'performance': {
                'expected_return': round(performance[0] * 100, 2),
                'volatility': round(performance[1] * 100, 2),
                'sharpe_ratio': round(performance[2], 2)
            },
            'error': None
        }
        
    except Exception as e:
        return {
            'error': f'Error optimizing portfolio: {str(e)}',
            'allocations': [],
            'performance': None
        }

def calculate_portfolio_metrics(holdings, total_worth_from_csv=None):
    # holdings: list of dicts [{"ticker": str, "amount": float/int, "price": float (optional)}]
    if not holdings or not isinstance(holdings, list):
        return {
            'total_value': 0,
            'expected_return': 0,
            'risk_level': 'N/A',
            'sector_exposure': {},
            'diversification_score': 0,
            'volatility': 0,
        }

    tickers = [h['ticker'] for h in holdings]
    amounts = {h['ticker']: float(h['amount']) for h in holdings}

    # Get prices: use CSV price if present, else yfinance
    prices = {}
    sectors = {}
    for h in holdings:
        ticker = h['ticker']
        price = h.get('price', None)
        if price is not None and price != '' and not pd.isna(price):
            try:
                prices[ticker] = float(price)
            except Exception:
                prices[ticker] = 0
        else:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                yf_price = info.get('regularMarketPrice')
                prices[ticker] = yf_price if yf_price is not None else 0
            except Exception:
                prices[ticker] = 0
        # Get sector info
        try:
            if 'sector' in h and h['sector']:
                sectors[ticker] = h['sector']
            else:
                stock = yf.Ticker(ticker)
                info = stock.info
                sectors[ticker] = info.get('sector', 'Other') or 'Other'
        except Exception:
            sectors[ticker] = 'Other'

    # Calculate total value
    total_value = sum(prices[t] * amounts[t] for t in tickers)
    # If total_worth_from_csv is provided and valid, use it
    if total_worth_from_csv is not None:
        try:
            total_value = float(total_worth_from_csv)
        except Exception:
            pass
    if total_value == 0:
        total_value = 1  # avoid division by zero

    # Calculate sector exposure
    sector_totals = {}
    for t in tickers:
        sector = sectors[t]
        sector_totals[sector] = sector_totals.get(sector, 0) + prices[t] * amounts[t]
    sector_exposure = {k: round(100 * v / total_value, 2) for k, v in sector_totals.items()}

    # Diversification score: number of sectors and tickers
    diversification_score = min(100, len(set(sectors.values())) * 15 + len(tickers) * 5)

    # Get historical data for risk/return
    data = pd.DataFrame()
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')
            if not hist.empty:
                data[ticker] = hist['Close']
        except Exception:
            continue
    if data.empty:
        # fallback
        return {
            'total_value': round(total_value, 2),
            'expected_return': 0,
            'risk_level': 'N/A',
            'sector_exposure': sector_exposure,
            'diversification_score': diversification_score,
            'volatility': 0,
        }

    # Portfolio weights by value
    weights = np.array([prices[t] * amounts[t] / total_value for t in tickers])
    # Calculate expected return and volatility
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    port_return = float(np.dot(weights, mu.loc[tickers])) * 100  # annualized %
    port_volatility = float(np.sqrt(np.dot(weights.T, np.dot(S.loc[tickers, tickers], weights)))) * 100  # annualized %

    # Risk level
    if port_volatility < 10:
        risk_level = 'Conservative'
    elif port_volatility < 18:
        risk_level = 'Moderate'
    else:
        risk_level = 'Aggressive'

    return {
        'total_value': round(total_value, 2),
        'expected_return': round(port_return, 2),
        'risk_level': risk_level,
        'sector_exposure': sector_exposure,
        'diversification_score': diversification_score,
        'volatility': round(port_volatility, 2),
    }