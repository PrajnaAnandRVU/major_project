# import yfinance as yf

# def get_fundamentals(ticker):
#     try:
#         stock = yf.Ticker(ticker)
#         info = stock.info
#         financials = stock.financials
        
#         if not info or 'shortName' not in info:
#             return {'error': 'No data found for this ticker'}
        
#         metrics = {
#             'ticker': ticker,
#             'company_name': info.get('shortName', 'N/A'),
#             'sector': info.get('sector', 'N/A'),
#             'market_cap': info.get('marketCap', 'N/A'),
#             'eps': info.get('trailingEps', 'N/A'),
#             'pe_ratio': info.get('trailingPE', 'N/A'),
#             'revenue': financials.loc['Total Revenue'][0] if 'Total Revenue' in financials.index else 'N/A',
#             'error': None
#         }
#         return metrics
#     except Exception as e:
#         return {'error': f'Error fetching fundamentals: {str(e)}'}

import yfinance as yf

def get_fundamentals(ticker):
    try:
        stock = yf.Ticker(ticker)

        info = stock.info
        financials = stock.financials
        earnings = stock.earnings

        if not info or 'shortName' not in info:
            return {'error': 'No data found for this ticker'}

        # ==============================
        # 📊 BASIC METRICS
        # ==============================
        metrics = {
            'ticker': ticker,
            'company_name': info.get('shortName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'eps': info.get('trailingEps', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
        }

        # ==============================
        # 📈 REVENUE TREND (BI Chart)
        # ==============================
        revenue_chart = []
        if 'Total Revenue' in financials.index:
            for date, value in financials.loc['Total Revenue'].items():
                revenue_chart.append({
                    "year": str(date.year),
                    "revenue": int(value)
                })

        # ==============================
        # 💰 NET INCOME TREND
        # ==============================
        profit_chart = []
        if 'Net Income' in financials.index:
            for date, value in financials.loc['Net Income'].items():
                profit_chart.append({
                    "year": str(date.year),
                    "net_income": int(value)
                })

        # ==============================
        # 📊 EPS TREND
        # ==============================
        eps_chart = []
        if earnings is not None:
            for year, row in earnings.iterrows():
                eps_chart.append({
                    "year": str(year),
                    "eps": float(row.get("Earnings", 0))
                })

        # ==============================
        # 📊 INSIGHTS (AUTO GENERATED)
        # ==============================
        insights = []

        if metrics['pe_ratio'] != 'N/A':
            if metrics['pe_ratio'] > 30:
                insights.append("High P/E ratio → potentially overvalued")
            elif metrics['pe_ratio'] < 15:
                insights.append("Low P/E ratio → potentially undervalued")

        if revenue_chart:
            if revenue_chart[0]['revenue'] > revenue_chart[-1]['revenue']:
                insights.append("Revenue declining over years ⚠️")
            else:
                insights.append("Revenue growing steadily 📈")

        if profit_chart:
            if profit_chart[0]['net_income'] > profit_chart[-1]['net_income']:
                insights.append("Profit declining ⚠️")
            else:
                insights.append("Profit increasing 💰")

        # ==============================
        # 📦 FINAL RESPONSE
        # ==============================
        return {
            "metrics": metrics,
            "charts": {
                "revenue": revenue_chart[::-1],   # oldest → newest
                "profit": profit_chart[::-1],
                "eps": eps_chart[::-1]
            },
            "insights": insights
        }

    except Exception as e:
        return {'error': f'Error fetching fundamentals: {str(e)}'}