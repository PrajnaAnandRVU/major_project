export interface StockData {
  metrics: {
    portfolio_value: number;
    portfolio_change: number;
    risk_score: number;
    risk_level: string;
    sharpe_ratio: number;
    sharpe_change: number;
    overall_sentiment: string;
    sentiment_strength: string;
    key_findings?: string[];
    recommendations?: string[];
  };
  news: NewsArticle[];
}

export interface NewsArticle {
  title: string;
  source: string;
  timestamp: string;
  content: string;
  url: string;
  sentiment: number;
  credibility: number;
}

export interface SentimentTermCount {
  term: string;
  article_count: number;
}

export interface SentimentArticleExplanation {
  title: string;
  sentiment: number;
  weighted_positive_hits: number;
  weighted_negative_hits: number;
  matched_positive_terms: string[];
  matched_negative_terms: string[];
  score_source: string;
  why: string;
}

export interface SentimentExplanation {
  method: string;
  description: string;
  thresholds: {
    bullish_above: number;
    bearish_below: number;
    neutral_between: [number, number];
  };
  label_rationale: string;
  aggregate: {
    article_count: number;
    mean_sentiment: number;
    sum_weighted_positive_hits: number;
    sum_weighted_negative_hits: number;
  };
  top_positive_terms: SentimentTermCount[];
  top_negative_terms: SentimentTermCount[];
  per_article: SentimentArticleExplanation[];
  uses_demo_news: boolean;
}

export interface SentimentAnalysis {
  overall_sentiment: {
    score: number;
    label: string;
  };
  sentiment_timeline: Array<{
    timestamp: string;
    score: number;
    label: string;
  }>;
  pnl_growth: Array<{
    year: string;
    revenue: number | null;
    net_income: number | null;
  }>;
  price_history_10y: Array<{
    timestamp: string;
    price: number;
  }>;
  key_findings: string[];
  recommendations: string[];
  top_news: NewsArticle[];
  explanation?: SentimentExplanation;
}

export interface PortfolioData {
  current_portfolio: {
    total_value: number;
    annualized_return: number;
    risk_level: string;
    asset_allocation: Array<{
      category: string;
      percentage: number;
    }>;
    top_holdings: Array<{
      ticker: string;
      company_name: string;
      amount: number;
      value: number;
      percentage: number;
    }>;
  };
  recommended_optimization: {
    projected_value: number;
    projected_return: number;
    optimized_risk_level: string;
    recommended_asset_allocation: Array<{
      category: string;
      percentage: number;
    }>;
    recommended_actions: Array<{
      action: string;
      ticker: string;
      company_name: string;
      details: string;
    }>;
    optimized_allocation: Array<{
      ticker: string;
      company_name: string;
      optimal_percentage: number;
    }>;
    tax_loss_harvesting: any[];
  };
  metrics: {
    expected_return: number;
    volatility: number;
    sharpe_ratio: number;
    total_value: number;
    allocations: Array<{
      ticker: string;
      weight: number;
    }>;
    risk_level: string;
    diversification_score: number;
    sector_exposure: Record<string, number>;
    recommendations: string[];
  };
  ai_analysis?: string;
  ai_suggestions?: Array<{
    ticker: string;
    company_name: string;
    sentiment: number;
    summary: string;
  }>;
}

export interface MarketData {
  metrics: {
    portfolio_value: number;
    portfolio_change: number;
    risk_score: number;
    risk_level: string;
    sharpe_ratio: number;
    sharpe_change: number;
    overall_sentiment: string;
    sentiment_strength: string;
  };
  news: NewsArticle[];
}

export interface SectorAnalysis {
  market_trend: string;
  spy_performance: number;
  market_outlook: string;
  top_sectors: Array<{
    name: string;
    performance: number;
  }>;
  timestamp: string;
} 