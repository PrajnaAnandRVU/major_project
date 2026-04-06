// The exported code uses Tailwind CSS. Install Tailwind CSS in your dev environment to ensure all styles work.
import React, { useState, useEffect, useCallback } from 'react';
import * as echarts from 'echarts';
import axios from 'axios';
import { useApiData } from './hooks/useApiData';
import { StockData, NewsArticle, SentimentAnalysis, PortfolioData, MarketData, SectorAnalysis } from './types';
import { fetchPortfolioData, fetchNewsData, fetchStockData, fetchSectorAnalysis, fetchFundamentalData } from './services/api';
import ReactMarkdown from 'react-markdown';
// @ts-ignore - remark-gfm may not have types
import remarkGfm from 'remark-gfm';

// API endpoint from environment variables
const API_BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:5001';

// Define a type for market chart data
interface MarketChartDataPoint {
  date: string;
  price: number;
}
interface MarketChart {
  symbol: string;
  name: string;
  data: MarketChartDataPoint[];
  current: number | null;
  change: number | null;
  ohlc: { open: number; high: number; low: number; close: number } | null;
}

// Helper to extract tickers from AI recommendations text
function extractSuggestedTickers(aiText: string, existingTickers: string[]): string[] {
  // Simple regex: look for uppercase words of 2-5 letters, not already in portfolio
  const tickerRegex = /\b([A-Z]{2,5})\b/g;
  const found = new Set<string>();
  let match;
  while ((match = tickerRegex.exec(aiText)) !== null) {
    const ticker = match[1];
    if (!existingTickers.includes(ticker) && ticker !== 'ETF' && ticker !== 'AI') {
      found.add(ticker);
    }
  }
  return Array.from(found);
}

const App: React.FC = () => {
  // Existing state
  const [darkMode, setDarkMode] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [showNotifications, setShowNotifications] = useState<boolean>(false);
  const [showProfileMenu, setShowProfileMenu] = useState<boolean>(false);
  const [showPreview, setShowPreview] = useState<boolean>(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  // New state for real-time data
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [newsData, setNewsData] = useState<NewsArticle[]>([]);
  const [stockData, setStockData] = useState<StockData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  /** Isolated from sentiment/portfolio `loading` so charts don't blank the rest of the UI */
  const [marketChartsLoading, setMarketChartsLoading] = useState<boolean>(false);
  const [marketChartsError, setMarketChartsError] = useState<string | null>(null);
  const [sentimentData, setSentimentData] = useState<SentimentAnalysis | null>(null);
  const [fundamentalData, setFundamentalData] = useState<any>(null);

  // Add API data hook
  const {
    loading: apiLoading,
    error: apiError,
    portfolioData: apiPortfolioData,
    newsData: apiNewsData,
    marketData: apiMarketData,
    sectorData: apiSectorData,
    getStockData,
    getSentimentAnalysis,
    uploadPortfolio,
    refreshData,
    getGeneralNews,
    getRecommendations,
  } = useApiData();

  // Add state for selected financial statement tab
  const [statementTab, setStatementTab] = useState<'income_statement' | 'balance_sheet' | 'cash_flow'>('income_statement');

  // Add state for recommendations
  const [recommendations, setRecommendations] = useState<any[]>([]);

  // Add state for high-impact news for recommendations
  const [recNews, setRecNews] = useState<Record<string, any>>({});

  // 1. Add state for market mini-charts
  const [marketCharts, setMarketCharts] = useState<MarketChart[]>([
    { symbol: 'AAPL', name: 'Apple Inc.', data: [], current: null, change: null, ohlc: null },
    { symbol: 'MSFT', name: 'Microsoft Corp.', data: [], current: null, change: null, ohlc: null },
    { symbol: 'TSLA', name: 'Tesla Inc.', data: [], current: null, change: null, ohlc: null },
  ]);

  // Add state for refresh button
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());

  // Add state for selected risk profile
  const [selectedRiskProfile, setSelectedRiskProfile] = useState<'Conservative' | 'Moderate' | 'Aggressive'>('Moderate');

  // Fetch news data (DASHBOARD: use general news)
  const handleNewsFetch = useCallback(async () => {
    try {
      const data = await getGeneralNews();
      setNewsData(data);
    } catch (err) {
      console.error('Error fetching general news:', err);
      setNewsData([]);
    }
  }, [getGeneralNews]);

  // Fetch stock data
  const handleStockFetch = useCallback(async (symbol: string) => {
    try {
      setLoading(true);
      const data = await fetchStockData(symbol);
      setStockData(data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch stock data');
      console.error('Error fetching stock data:', err);
      setStockData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch sentiment analysis
  const handleSentimentAnalysis = useCallback(async (ticker: string) => {
    if (!ticker || !ticker.trim()) {
      return;
    }
    try {
      setLoading(true);
      // Clear old sentiment data immediately to prevent stale data display
      setSentimentData(null);
      const symbol = ticker.trim().toUpperCase();
      console.log(`Fetching sentiment analysis for ticker: ${symbol}`);
      const data: SentimentAnalysis = await getSentimentAnalysis(symbol);
      console.log(`Sentiment analysis data received for ${symbol}:`, data);
      console.log(`Overall sentiment score for ${symbol}:`, data?.overall_sentiment?.score);
      setSentimentData(data);
    } catch (err) {
      console.error('Error analyzing sentiment:', err);
      setSentimentData(null);
    } finally {
      setLoading(false);
    }
  }, [getSentimentAnalysis]);

  // Fetch fundamental analysis
  const handleFundamentalAnalysis = useCallback(async (ticker: string) => {
    try {
      setLoading(true);
      const data = await fetchFundamentalData(ticker);
      setFundamentalData(data);
    } catch (err) {
      console.error('Error fetching fundamental data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial data fetch
  useEffect(() => {
    handleNewsFetch();
    // Fetch sector analysis
    fetchSectorAnalysis().then(data => {
      // Update sector allocation chart
      const sectorAllocationElement = document.getElementById('sector-allocation');
      if (sectorAllocationElement) {
        const sectorChart = echarts.init(sectorAllocationElement);
        const sectorData = data.top_sectors.map((sector: { name: string; performance: number }) => ({ // Explicitly type sector
          name: sector.name,
          value: sector.performance
        }));
        
        const sectorOption = {
          animation: false,
          tooltip: {
            trigger: 'item',
            formatter: '{a} <br/>{b}: {c} ({d}%)'
          },
          legend: {
            orient: 'vertical',
            right: 10,
            top: 'center',
            data: sectorData.map((item: { name: string }) => item.name) // Explicitly type item
          },
          series: [{
            name: 'Sector Allocation',
            type: 'pie',
            radius: ['50%', '70%'],
            avoidLabelOverlap: false,
            label: {
              show: false,
              position: 'center'
            },
            emphasis: {
              label: {
                show: true,
                fontSize: '14',
                fontWeight: 'bold'
              }
            },
            labelLine: {
              show: false
            },
            data: sectorData
          }]
        };
        sectorChart.setOption(sectorOption);
      }
    }).catch(err => {
      console.error('Error fetching sector analysis:', err);
    });
  }, [handleNewsFetch]);

  // Real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      handleNewsFetch();
      fetchSectorAnalysis().catch(err => {
        console.error('Error fetching sector analysis:', err);
      });
    }, 60000); // Update every minute

    return () => clearInterval(interval);
  }, [handleNewsFetch]);

  // Add ECharts gauge for overall sentiment
  useEffect(() => {
    if (sentimentData?.overall_sentiment && activeTab === 'sentiment') {
      // Use setTimeout to ensure DOM is ready
      const timer = setTimeout(() => {
        const gaugeElement = document.getElementById('overall-sentiment-gauge');
        if (gaugeElement) {
          // Dispose existing chart if any
          const existingChart = echarts.getInstanceByDom(gaugeElement);
          if (existingChart) {
            existingChart.dispose();
          }
          
          const gaugeChart = echarts.init(gaugeElement);
          const score = sentimentData.overall_sentiment.score || 0;
          let color = '#6B7280'; // gray
          if (score > 0.1) color = '#10B981'; // green
          else if (score < -0.1) color = '#EF4444'; // red
          const gaugeOption = {
            series: [
              {
                type: 'gauge',
                startAngle: 210,
                endAngle: -30,
                min: -1,
                max: 1,
                splitNumber: 4,
                axisLine: {
                  lineStyle: {
                    width: 18,
                    color: [
                      [0.5, color],
                      [1, '#E5E7EB']
                    ]
                  }
                },
                pointer: {
                  show: true,
                  length: '70%',
                  width: 6
                },
                progress: {
                  show: true,
                  width: 18,
                  itemStyle: {
                    color: color
                  }
                },
                axisTick: {
                  show: false
                },
                splitLine: {
                  show: false
                },
                axisLabel: {
                  show: false
                },
                detail: {
                  valueAnimation: true,
                  fontSize: 36,
                  fontWeight: 'bold',
                  color: color,
                  offsetCenter: [0, '30%'],
                  formatter: function(value: number) {
                    return value.toFixed(2);
                  }
                },
                data: [
                  {
                    value: score
                  }
                ]
              }
            ]
          };
          gaugeChart.setOption(gaugeOption);
          
          const handleResize = () => gaugeChart.resize();
          window.addEventListener('resize', handleResize);
        }
      }, 100);
      
      return () => {
        clearTimeout(timer);
        const gaugeElement = document.getElementById('overall-sentiment-gauge');
        if (gaugeElement) {
          const chart = echarts.getInstanceByDom(gaugeElement);
          if (chart) {
            chart.dispose();
          }
        }
      };
    }
  }, [sentimentData?.overall_sentiment, activeTab]);

  // Add ECharts line chart for 10-year price history
  useEffect(() => {
    if (sentimentData?.price_history_10y && sentimentData.price_history_10y.length > 0 && activeTab === 'sentiment') {
      const timer = setTimeout(() => {
        const chartElement = document.getElementById('price-history-10y-chart');
        if (chartElement) {
          // Dispose existing chart if any
          const existingChart = echarts.getInstanceByDom(chartElement);
          if (existingChart) {
            existingChart.dispose();
          }
          
          const chart = echarts.init(chartElement);
          
          // Filter data to the last 12 months
          const oneYearAgo = new Date();
          oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);
          
          // Filter data to last 12 months (timestamps are in "MMM YYYY" format)
          const lastYearData = sentimentData.price_history_10y.slice(-12);

        const months = lastYearData.map((item: any) => item.timestamp);
        const prices = lastYearData.map((item: any) => item.price);
        const option = {
          tooltip: {
            trigger: 'axis',
            formatter: (params: any) => {
              let str = `<b>Month: ${params[0].axisValue}</b><br/>`;
              params.forEach((p: any) => {
                str += `${p.marker} ${p.seriesName}: $${p.value.toLocaleString()}<br/>`;
              });
              return str;
            }
          },
          legend: {
            data: ['Price']
          },
          grid: {
            left: '3%',
            right: '4%',
            bottom: '15%',
            containLabel: true
          },
          xAxis: {
            type: 'category',
            data: months,
            axisLabel: {
              show: true,
              rotate: 45,
              interval: 0,
              formatter: (value: string) => value
            }
          },
          yAxis: {
            type: 'value',
            axisLabel: {
              formatter: function (value: number) {
                if (value >= 1e3) return `$${(value / 1e3).toFixed(1)}k`;
                return `$${value}`;
              }
            }
          },
          series: [
            {
              name: 'Price',
              type: 'line',
              data: prices,
              smooth: true,
              lineStyle: { color: '#6366F1', width: 3 },
              itemStyle: { color: '#6366F1' }
            }
          ]
        };
        chart.setOption(option);
        
        const handleResize = () => chart.resize();
        window.addEventListener('resize', handleResize);
      }
      }, 100);
      
      return () => {
        clearTimeout(timer);
        const chartElement = document.getElementById('price-history-10y-chart');
        if (chartElement) {
          const chart = echarts.getInstanceByDom(chartElement);
          if (chart) {
            chart.dispose();
          }
        }
      };
    }
  }, [sentimentData?.price_history_10y, activeTab]);

  // Add ECharts line chart for sentiment timeline
  useEffect(() => {
    if (sentimentData?.sentiment_timeline && sentimentData.sentiment_timeline.length > 0 && activeTab === 'sentiment') {
      const timer = setTimeout(() => {
        const chartElement = document.getElementById('sentiment-chart');
        if (chartElement) {
          // Dispose existing chart if any
          const existingChart = echarts.getInstanceByDom(chartElement);
          if (existingChart) {
            existingChart.dispose();
          }
          
          const chart = echarts.init(chartElement);
        
        // Take last 24 months for better visualization
        const timelineData = sentimentData.sentiment_timeline.slice(-24);
        
        const timestamps = timelineData.map((item) => item.timestamp);
        const scores = timelineData.map((item) => item.score);
        
        const option = {
          animation: false,
          tooltip: {
            trigger: 'axis',
            formatter: (params: any) => {
              const p = params[0];
              return `<b>${p.axisValue}</b><br/>${p.marker} Sentiment: ${p.value.toFixed(2)}`;
            }
          },
          grid: {
            left: '3%',
            right: '4%',
            bottom: '15%',
            containLabel: true
          },
          xAxis: {
            type: 'category',
            data: timestamps,
            axisLabel: {
              show: true,
              rotate: 45,
              interval: Math.floor(timestamps.length / 12), // Show ~12 labels
              formatter: (value: string) => value
            }
          },
          yAxis: {
            type: 'value',
            min: -1,
            max: 1,
            axisLabel: {
              formatter: (value: number) => value.toFixed(1)
            }
          },
          series: [{
            name: 'Sentiment Score',
            type: 'line',
            data: scores,
            smooth: true,
            lineStyle: {
              width: 2
            },
            itemStyle: {
              color: function(params: any) {
                const score = params.value;
                if (score > 0.1) return '#10B981'; // green
                if (score < -0.1) return '#EF4444'; // red
                return '#6B7280'; // gray
              }
            },
            areaStyle: {
              color: {
                type: 'linear',
                x: 0,
                y: 0,
                x2: 0,
                y2: 1,
                colorStops: [{
                  offset: 0,
                  color: 'rgba(16, 185, 129, 0.3)'
                }, {
                  offset: 1,
                  color: 'rgba(239, 68, 68, 0.3)'
                }]
              }
            }
          }]
        };
        chart.setOption(option);
        
        const handleResize = () => chart.resize();
        window.addEventListener('resize', handleResize);
      }
      }, 100);
      
      return () => {
        clearTimeout(timer);
        const chartElement = document.getElementById('sentiment-chart');
        if (chartElement) {
          const chart = echarts.getInstanceByDom(chartElement);
          if (chart) {
            chart.dispose();
          }
        }
      };
    }
  }, [sentimentData?.sentiment_timeline, activeTab]);

  // Fetch recommendations for dashboard
  useEffect(() => {
    if (activeTab === 'dashboard') {
      console.log('Dashboard tab active, fetching recommendations...');
      // Clear previous recommendations to show loading state
      setRecommendations([]);
      getRecommendations()
        .then(data => {
          console.log('Setting recommendations:', data);
          if (Array.isArray(data) && data.length > 0) {
            setRecommendations(data);
          } else {
            console.warn('No recommendations data received or empty array');
            setRecommendations([]);
          }
        })
        .catch(err => {
          console.error('Error fetching recommendations:', err);
          setRecommendations([]);
        });
    } else {
      // Clear recommendations when switching away from dashboard
      setRecommendations([]);
    }
  }, [activeTab, getRecommendations]);

  // Fetch high-impact news for top recommendations
  useEffect(() => {
    async function fetchRecNews() {
      const newsMap: Record<string, any> = {};
      for (const rec of recommendations) {
        // Fetch news for each recommended stock
        try {
          const res = await fetchStockData(rec.ticker);
          // Find the most impactful news (e.g., highest sentiment or credibility)
          const highImpact = res.news?.sort((a: any, b: any) => (b.credibility || 0) - (a.credibility || 0))[0];
          if (highImpact) newsMap[rec.ticker] = highImpact;
        } catch (e) {
          // Ignore errors for individual stocks
        }
      }
      setRecNews(newsMap);
    }
    if (recommendations.length > 0) fetchRecNews();
  }, [recommendations]);

  // 2. Fetch historical price data for each ticker (Alpha Vantage via backend endpoint)
  useEffect(() => {
    fetchMarketChartData();
  }, []);

  // 3. Render three market OHLC cards stacked vertically
  const renderMarketCharts = () => {
    if (marketChartsLoading) {
      return (
        <div className="flex flex-col gap-4 mb-6">
          {[1, 2, 3].map((i) => (
            <div key={i} className="p-6 rounded-lg shadow-md bg-white animate-pulse">
              <div className="h-6 bg-gray-200 rounded w-1/4 mb-4"></div>
              <div className="space-y-3">
                <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                <div className="h-4 bg-gray-200 rounded w-1/2"></div>
              </div>
            </div>
          ))}
        </div>
      );
    }

    if (marketChartsError) {
      return (
        <div className="p-6 rounded-lg shadow-md bg-red-50 text-red-700 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <i className="fas fa-exclamation-circle mr-2"></i>
              <span>Error loading market data: {marketChartsError}</span>
            </div>
            <button
              onClick={() => fetchMarketChartData()}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
            >
              Retry
            </button>
          </div>
        </div>
      );
    }

    return (
      <div className="flex flex-col gap-4 mb-6">
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-xl font-bold">Markets</h2>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-500">
              Last updated: {lastRefresh.toLocaleTimeString()}
            </span>
            <button
              onClick={() => fetchMarketChartData()}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              disabled={marketChartsLoading}
            >
              {marketChartsLoading ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
        </div>
        {marketCharts.map((m: any, idx: number) => (
          <div key={m.symbol} className="p-6 rounded-lg shadow-md bg-white flex flex-col items-start">
            <div className="flex items-center mb-2">
              <span className="font-bold text-lg mr-2">{m.name}</span>
              <span className="text-xs text-gray-400">({m.symbol})</span>
            </div>
            {m.ohlc ? (
              <div className="w-full">
                <div className="flex flex-wrap gap-4 mb-2">
                  <span className="text-sm">Open: <span className="font-semibold">${m.ohlc.open.toFixed(2)}</span></span>
                  <span className="text-sm">High: <span className="font-semibold">${m.ohlc.high.toFixed(2)}</span></span>
                  <span className="text-sm">Low: <span className="font-semibold">${m.ohlc.low.toFixed(2)}</span></span>
                  <span className="text-sm">Close: <span className="font-semibold">${m.ohlc.close.toFixed(2)}</span></span>
                </div>
                <div className="flex items-center">
                  <span className="text-2xl font-bold">${m.ohlc.close.toFixed(2)}</span>
                  {m.change !== null && (
                    <span className={`ml-2 text-lg font-semibold ${m.change > 0 ? 'text-green-500' : m.change < 0 ? 'text-red-500' : 'text-gray-500'}`}>
                      {m.change > 0 ? '+' : ''}{m.change.toFixed(2)}%
                    </span>
                  )}
                </div>
              </div>
            ) : (
              <span className="text-gray-500">No data available</span>
            )}
          </div>
        ))}
      </div>
    );
  };

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
  };

  const toggleNotifications = () => {
    setShowNotifications(!showNotifications);
    if (showNotifications) setShowProfileMenu(false);
  };

  const toggleProfileMenu = () => {
    setShowProfileMenu(!showProfileMenu);
    if (showProfileMenu) setShowNotifications(false);
  };

  // Update the metrics cards to use real data
  const renderMetricsCards = () => {
    if (!portfolioData) return null;

    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
          <div className="flex justify-between items-start">
            <div>
              <p className="text-sm text-gray-500">Portfolio Value</p>
              <h3 className="text-2xl font-bold">${portfolioData.current_portfolio.total_value.toLocaleString()}</h3>
              <p className="text-sm text-green-500 flex items-center mt-1">
                <i className="fas fa-arrow-up mr-1"></i>
                {portfolioData.current_portfolio.annualized_return > 0 ? '+' : ''}
                {portfolioData.current_portfolio.annualized_return.toFixed(1)}%
              </p>
            </div>
            <div className={`p-3 rounded-full ${darkMode ? 'bg-blue-900' : 'bg-blue-100'}`}>
              <i className="fas fa-dollar-sign text-blue-600"></i>
            </div>
          </div>
        </div>
        <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
          <div className="flex justify-between items-start">
            <div>
              <p className="text-sm text-gray-500">Risk Score</p>
              <h3 className="text-2xl font-bold">{portfolioData.metrics.risk_level}</h3>
              <p className="text-sm text-yellow-500 flex items-center mt-1">
                <i className="fas fa-exclamation-triangle mr-1"></i>
                {portfolioData.metrics.risk_level}
              </p>
            </div>
            <div className={`p-3 rounded-full ${darkMode ? 'bg-yellow-900' : 'bg-yellow-100'}`}>
              <i className="fas fa-shield-alt text-yellow-600"></i>
            </div>
          </div>
        </div>
        <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
          <div className="flex justify-between items-start">
            <div>
              <p className="text-sm text-gray-500">Sharpe Ratio</p>
              <h3 className="text-2xl font-bold">{portfolioData.metrics.sharpe_ratio.toFixed(2)}</h3>
              <p className="text-sm text-green-500 flex items-center mt-1">
                <i className="fas fa-arrow-up mr-1"></i>
                {portfolioData.metrics.expected_return.toFixed(2)}
              </p>
            </div>
            <div className={`p-3 rounded-full ${darkMode ? 'bg-green-900' : 'bg-green-100'}`}>
              <i className="fas fa-chart-bar text-green-600"></i>
            </div>
          </div>
        </div>
        <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
          <div className="flex justify-between items-start">
            <div>
              <p className="text-sm text-gray-500">Overall Sentiment</p>
              <h3 className="text-2xl font-bold">{portfolioData.metrics.recommendations[0] || 'Neutral'}</h3>
              <p className="text-sm text-green-500 flex items-center mt-1">
                <i className="fas fa-arrow-up mr-1"></i>
                Strong
              </p>
            </div>
            <div className={`p-3 rounded-full ${darkMode ? 'bg-green-900' : 'bg-green-100'}`}>
              <i className="fas fa-thumbs-up text-green-600"></i>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Update the news feed to use real data
  const renderNewsFeed = () => {
    if (apiLoading) {
      return (
        <div className={`mt-6 p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
          <h2 className="text-xl font-bold mb-4">Latest Financial News</h2>
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="p-4 rounded-lg animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2 mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-full"></div>
              </div>
            ))}
          </div>
        </div>
      );
    }

    if (apiError && (!newsData || newsData.length === 0)) {
      return (
        <div className={`mt-6 p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
          <h2 className="text-xl font-bold mb-4">Latest Financial News</h2>
          <div className="p-4 rounded-lg bg-red-50 text-red-700">
            <p>Error loading news: {apiError}</p>
            <button
              type="button"
              onClick={() => {
                refreshData();
                handleNewsFetch();
              }}
              className="mt-2 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
            >
              Retry
            </button>
          </div>
        </div>
      );
    }

    if (!newsData || newsData.length === 0) {
      return (
        <div className={`mt-6 p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
          <h2 className="text-xl font-bold mb-4">Latest Financial News</h2>
          <div className="p-4 rounded-lg bg-gray-50 text-gray-700">
            <p>No news available at the moment.</p>
            <button
              onClick={handleNewsFetch}
              className="mt-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Refresh
            </button>
          </div>
        </div>
      );
    }

    return (
      <div className={`mt-6 p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-bold">Latest Financial News</h2>
          <button
            onClick={handleNewsFetch}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            disabled={apiLoading}
          >
            {apiLoading ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
        <div className="space-y-4">
          {newsData.map((article, index) => (
            <div
              key={index}
              className={`p-4 rounded-lg ${
                article.sentiment > 0
                  ? darkMode ? 'bg-green-900 bg-opacity-20' : 'bg-green-50'
                  : article.sentiment < 0
                  ? darkMode ? 'bg-red-900 bg-opacity-20' : 'bg-red-50'
                  : darkMode ? 'bg-gray-700 bg-opacity-20' : 'bg-gray-100'
              }`}
            >
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="font-bold">{article.title}</h3>
                  <p className="text-sm text-gray-500">{article.source} • {article.timestamp}</p>
                </div>
                <div className={`px-2 py-1 rounded text-xs ${
                  article.sentiment > 0
                    ? darkMode ? 'bg-green-900 text-green-400' : 'bg-green-100 text-green-800'
                    : article.sentiment < 0
                    ? darkMode ? 'bg-red-900 text-red-400' : 'bg-red-100 text-red-800'
                    : darkMode ? 'bg-gray-600 text-gray-300' : 'bg-gray-200 text-gray-700'
                }`}>
                  {article.sentiment > 0 ? 'Bullish' : article.sentiment < 0 ? 'Bearish' : 'Neutral'}
                </div>
              </div>
              <p className="text-sm mt-2">{article.content}</p>
              <div className="mt-2 flex items-center">
                <span className="text-xs text-gray-500">Credibility:</span>
                <div className="ml-2 flex">
                  {[...Array(5)].map((_, i) => (
                    <span
                      key={i}
                      className={`text-xs ${
                        i < article.credibility
                          ? darkMode ? 'text-yellow-400' : 'text-yellow-500'
                          : darkMode ? 'text-gray-600' : 'text-gray-300'
                      }`}
                    >
                      ★
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Add this useEffect after the other useEffects
  useEffect(() => {
    // Auto-analyze AAPL when sentiment tab is first opened
    if (activeTab === 'sentiment' && !sentimentData && !loading && searchQuery.trim() === '') {
      handleSentimentAnalysis('AAPL');
      setSearchQuery('AAPL');
    }
  }, [activeTab, sentimentData, loading, handleSentimentAnalysis, searchQuery]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    // CSV validation
    const text = await file.text();
    const lines = text.split(/\r?\n/).filter(Boolean);
    if (lines.length < 2) {
      alert('CSV file is empty or missing data rows.');
      return;
    }
    const header = lines[0].split(',').map(h => h.trim().toLowerCase());
    const required = ['ticker', 'amount', 'price'];
    const missing = required.filter(col => !header.includes(col));
    if (missing.length > 0) {
      alert(`CSV is missing required columns: ${missing.join(', ')}`);
      return;
    }
    setUploadedFile(file); // <-- Set uploadedFile so risk profile buttons work
    try {
      setLoading(true);
      const formData = new FormData();
      formData.append('portfolio', file);
      formData.append('risk_tolerance', 'Moderate'); // Default risk tolerance
      const response = await axios.post(`${API_BASE_URL}/portfolio`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setPortfolioData(response.data);
      // Show success message
      const successMessage = document.createElement('div');
      successMessage.className = `fixed bottom-4 right-4 px-6 py-3 rounded-lg ${darkMode ? 'bg-green-900' : 'bg-green-100'} text-green-700 z-50`;
      successMessage.textContent = 'Portfolio file uploaded successfully!';
      document.body.appendChild(successMessage);
      setTimeout(() => {
        document.body.removeChild(successMessage);
      }, 3000);
    } catch (error) {
      console.error('Error uploading portfolio:', error);
      // Show error message
      let message = 'Error uploading portfolio file';
      if (error && typeof error === 'object' && 'code' in error && error.code === 'ECONNABORTED') {
        message = 'Portfolio optimization is taking too long. Please try again later or with a smaller file.';
      } else if (error instanceof Error) {
        message = error.message;
      }
      const errorMessage = document.createElement('div');
      errorMessage.className = `fixed bottom-4 right-4 px-6 py-3 rounded-lg ${darkMode ? 'bg-red-900' : 'bg-red-100'} text-red-700 z-50`;
      errorMessage.textContent = message;
      document.body.appendChild(errorMessage);
      setTimeout(() => {
        document.body.removeChild(errorMessage);
      }, 3000);
    } finally {
      setLoading(false);
    }
  };

  const handleRiskProfileChange = async (riskProfile: 'Conservative' | 'Moderate' | 'Aggressive') => {
    if (!uploadedFile) return;
    setSelectedRiskProfile(riskProfile); // <-- Visually mark selected
    setLoading(true); // Show loading indicator while fetching new recommendations
    try {
      const formData = new FormData();
      formData.append('portfolio', uploadedFile);
      formData.append('risk_tolerance', riskProfile);

      const response = await axios.post(`${API_BASE_URL}/portfolio`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setPortfolioData(response.data); // This will update ai_analysis and all portfolio data
    } catch (error) {
      console.error('Error updating risk profile:', error);
      // Handle error appropriately
    }
    setLoading(false); // Hide loading indicator
  };

  // Add fetchMarketChartData function inside App component
  const fetchMarketChartData = async () => {
    setMarketChartsLoading(true);
    setMarketChartsError(null);
    try {
      const tickers = [
        { symbol: 'AAPL', name: 'Apple Inc.' },
        { symbol: 'MSFT', name: 'Microsoft Corp.' },
        { symbol: 'TSLA', name: 'Tesla Inc.' },
      ];
      
      // Fetch one at a time to avoid rate limits
      const results: MarketChart[] = [];
      for (const t of tickers) {
        try {
          const resp = await fetch(`${API_BASE_URL}/api/polygon_history?ticker=${t.symbol}`);
          if (!resp.ok) {
            const errorData = await resp.json();
            console.error(`Error fetching data for ${t.symbol}:`, errorData);
            if (resp.status === 429) {
              // If rate limited, wait and retry
              await new Promise(resolve => setTimeout(resolve, 5000));
              const retryResp = await fetch(`${API_BASE_URL}/api/polygon_history?ticker=${t.symbol}`);
              if (!retryResp.ok) {
                throw new Error(`HTTP error! status: ${retryResp.status}, message: ${errorData.error || 'Unknown error'}`);
              }
              const retryJson = await retryResp.json();
              const history = retryJson.history || [];
              if (history.length > 0) {
                const latest = history[history.length - 1];
                const prev = history.length > 1 ? history[history.length - 2] : null;
                const percentChange = latest && prev ? ((latest.close - prev.close) / prev.close) * 100 : null;
                results.push({
                  symbol: t.symbol,
                  name: t.name,
                  data: history,
                  current: latest.close,
                  change: percentChange,
                  ohlc: { 
                    open: latest.open, 
                    high: latest.high, 
                    low: latest.low, 
                    close: latest.close 
                  }
                });
              }
            } else {
              throw new Error(`HTTP error! status: ${resp.status}, message: ${errorData.error || 'Unknown error'}`);
            }
          } else {
            const json = await resp.json();
            const history = json.history || [];
            if (history.length > 0) {
              const latest = history[history.length - 1];
              const prev = history.length > 1 ? history[history.length - 2] : null;
              const percentChange = latest && prev ? ((latest.close - prev.close) / prev.close) * 100 : null;
              results.push({
                symbol: t.symbol,
                name: t.name,
                data: history,
                current: latest.close,
                change: percentChange,
                ohlc: { 
                  open: latest.open, 
                  high: latest.high, 
                  low: latest.low, 
                  close: latest.close 
                }
              });
            }
          }
        } catch (e) {
          console.error(`Error fetching data for ${t.symbol}:`, e);
          results.push({ symbol: t.symbol, name: t.name, data: [], current: null, change: null, ohlc: null });
        }
        // Add a small delay between requests
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      setMarketCharts(results);
      setLastRefresh(new Date());
    } catch (e) {
      console.error('Error fetching market data:', e);
      setMarketChartsError(e instanceof Error ? e.message : 'Failed to fetch market data');
    } finally {
      setMarketChartsLoading(false);
    }
  };

  return (
    <div>
      <div className={`min-h-screen ${darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'}`}>
        {/* Header */}
        <header className={`fixed w-full z-10 ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-md`} style={{marginTop: 0, paddingTop: 0}}>
          <div className="container mx-auto px-4 py-3 flex items-center justify-between">
            <div className="flex items-center">
              <div className="text-2xl font-bold text-blue-600">
                <i className="fas fa-chart-line mr-2"></i>
                Finsense AI
              </div>
            </div>
            <div className="hidden md:flex items-center space-x-6">
              <button
                onClick={() => handleTabChange('dashboard')}
                className={`px-3 py-2 cursor-pointer whitespace-nowrap ${activeTab === 'dashboard' ? 'text-blue-600 border-b-2 border-blue-600' : ''}`}
              >
                Dashboard
              </button>
              <button
                onClick={() => handleTabChange('sentiment')}
                className={`px-3 py-2 cursor-pointer whitespace-nowrap ${activeTab === 'sentiment' ? 'text-blue-600 border-b-2 border-blue-600' : ''}`}
              >
                Sentiment Analysis
              </button>
              <button
                onClick={() => handleTabChange('fundamental')}
                className={`px-3 py-2 cursor-pointer whitespace-nowrap ${activeTab === 'fundamental' ? 'text-blue-600 border-b-2 border-blue-600' : ''}`}
              >
                Fundamental Analysis
              </button>
              <button
                onClick={() => handleTabChange('portfolio')}
                className={`px-3 py-2 cursor-pointer whitespace-nowrap ${activeTab === 'portfolio' ? 'text-blue-600 border-b-2 border-blue-600' : ''}`}
              >
                Portfolio Optimization
              </button>
            </div>
            <div className="flex items-center space-x-4">
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search stocks..."
                  className={`px-4 py-2 rounded-full text-sm border-none focus:outline-none focus:ring-2 focus:ring-blue-500 ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
                <button className="absolute right-3 top-1/2 transform -translate-y-1/2 cursor-pointer">
                  <i className="fas fa-search text-gray-400"></i>
                </button>
              </div>
              <button
                onClick={toggleDarkMode}
                className="p-2 rounded-full cursor-pointer !rounded-button"
              >
                {darkMode ? (
                  <i className="fas fa-sun text-yellow-400"></i>
                ) : (
                  <i className="fas fa-moon text-gray-600"></i>
                )}
              </button>
              <div className="relative">
                <button
                  onClick={toggleNotifications}
                  className="p-2 rounded-full cursor-pointer !rounded-button"
                >
                  <i className="fas fa-bell"></i>
                  <span className="absolute top-0 right-0 h-2 w-2 rounded-full bg-red-500"></span>
                </button>
                {showNotifications && (
                  <div className={`absolute right-0 mt-2 w-80 ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-md shadow-lg py-1 z-20`}>
                    <div className="px-4 py-2 border-b border-gray-200">
                      <h3 className="text-lg font-semibold">Notifications</h3>
                    </div>
                    <div className="max-h-96 overflow-y-auto">
                      <div className={`px-4 py-3 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'} cursor-pointer hover:bg-gray-100 hover:bg-opacity-20`}>
                        <p className="text-sm font-medium">AAPL sentiment turned bullish</p>
                        <p className="text-xs text-gray-500">2 hours ago</p>
                      </div>
                      <div className={`px-4 py-3 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'} cursor-pointer hover:bg-gray-100 hover:bg-opacity-20`}>
                        <p className="text-sm font-medium">Portfolio rebalancing suggested</p>
                        <p className="text-xs text-gray-500">Yesterday</p>
                      </div>
                      <div className={`px-4 py-3 cursor-pointer hover:bg-gray-100 hover:bg-opacity-20`}>
                        <p className="text-sm font-medium">New sector analysis available</p>
                        <p className="text-xs text-gray-500">May 17, 2025</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              <div className="relative">
                <button
                  onClick={toggleProfileMenu}
                  className="flex items-center cursor-pointer !rounded-button"
                >
                  <div className="h-8 w-8 rounded-full bg-blue-500 flex items-center justify-center text-white">
                    <span className="text-sm font-medium">JD</span>
                  </div>
                </button>
                {showProfileMenu && (
                  <div className={`absolute right-0 mt-2 w-48 ${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-md shadow-lg py-1 z-20`}>
                    <a href="#" className="block px-4 py-2 text-sm hover:bg-gray-100 hover:bg-opacity-20">Profile</a>
                    <a href="#" className="block px-4 py-2 text-sm hover:bg-gray-100 hover:bg-opacity-20">Settings</a>
                    <a href="#" className="block px-4 py-2 text-sm hover:bg-gray-100 hover:bg-opacity-20">Sign out</a>
                  </div>
                )}
              </div>
            </div>
          </div>
        </header>
        {/* Mobile Navigation */}
        <div className={`md:hidden fixed bottom-0 w-full z-10 ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-lg`}>
          <div className="flex justify-around py-2">
            <button
              onClick={() => handleTabChange('dashboard')}
              className={`p-2 flex flex-col items-center cursor-pointer whitespace-nowrap ${activeTab === 'dashboard' ? 'text-blue-600' : ''}`}
            >
              <i className="fas fa-chart-line text-lg"></i>
              <span className="text-xs mt-1">Dashboard</span>
            </button>
            <button
              onClick={() => handleTabChange('sentiment')}
              className={`p-2 flex flex-col items-center cursor-pointer whitespace-nowrap ${activeTab === 'sentiment' ? 'text-blue-600' : ''}`}
            >
              <i className="fas fa-comments text-lg"></i>
              <span className="text-xs mt-1">Sentiment</span>
            </button>
            <button
              onClick={() => handleTabChange('fundamental')}
              className={`p-2 flex flex-col items-center cursor-pointer whitespace-nowrap ${activeTab === 'fundamental' ? 'text-blue-600' : ''}`}
            >
              <i className="fas fa-file-invoice-dollar text-lg"></i>
              <span className="text-xs mt-1">Fundamental</span>
            </button>
            <button
              onClick={() => handleTabChange('portfolio')}
              className={`p-2 flex flex-col items-center cursor-pointer whitespace-nowrap ${activeTab === 'portfolio' ? 'text-blue-600' : ''}`}
            >
              <i className="fas fa-briefcase text-lg"></i>
              <span className="text-xs mt-1">Portfolio</span>
            </button>
          </div>
        </div>
        {/* Main Content */}
        <main className="pt-12 pb-16 md:pb-0 min-h-screen">
          {loading && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <div className="bg-white rounded-lg p-6 flex items-center space-x-4">
                <div className="animate-spin rounded-full h-8 w-8 border-4 border-blue-600 border-t-transparent"></div>
                <span>Loading data...</span>
              </div>
            </div>
          )}

          {error && (
            <div className="fixed bottom-4 right-4 px-6 py-3 rounded-lg bg-red-100 text-red-700 z-50">
              {error}
            </div>
          )}

          {activeTab === 'dashboard' && (
            <div className="container mx-auto px-4 py-6">
              {renderMetricsCards()}
              {/* Portfolio Chart and Sentiment Recommendations */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                <div className="lg:col-span-2 p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}">
                  <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-bold">Markets</h2>
                  </div>
                  {renderMarketCharts()}
                </div>
                <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}> 
                  <h2 className="text-xl font-bold mb-4">Sentiment-Driven Recommendations</h2>
                  <div className="space-y-4">
                    {recommendations && recommendations.length > 0 ? (
                      recommendations.map((rec, idx) => (
                        <div
                          key={rec.ticker || idx}
                          className={`p-4 rounded-lg border-l-4 ${rec.sentiment > 0 ? (darkMode ? 'bg-green-900 bg-opacity-20 border-green-500' : 'bg-green-50 border-green-500') : rec.sentiment < 0 ? (darkMode ? 'bg-red-900 bg-opacity-20 border-red-500' : 'bg-red-50 border-red-500') : (darkMode ? 'bg-gray-700 bg-opacity-20 border-gray-400' : 'bg-gray-100 border-gray-400')}`}
                        >
                          <div className="flex justify-between items-start">
                            <div>
                              <h3 className="font-bold">{rec.ticker || 'N/A'}</h3>
                              <p className="text-sm text-gray-500">{rec.company_name || rec.ticker || 'Unknown'}</p>
                            </div>
                            <div className="flex items-center">
                              <span className={`font-bold mr-2 ${rec.sentiment > 0 ? 'text-green-500' : rec.sentiment < 0 ? 'text-red-500' : 'text-gray-500'}`}>
                                {rec.sentiment !== undefined && rec.sentiment !== null 
                                  ? (rec.sentiment > 0 ? `+${rec.sentiment.toFixed(2)}` : rec.sentiment.toFixed(2))
                                  : '0.00'}
                              </span>
                              {rec.sentiment > 0 ? (
                                <i className="fas fa-arrow-up text-green-500"></i>
                              ) : rec.sentiment < 0 ? (
                                <i className="fas fa-arrow-down text-red-500"></i>
                              ) : null}
                            </div>
                          </div>
                          <p className="text-sm mt-2">{rec.summary || 'Analysis in progress...'}</p>
                        </div>
                      ))
                    ) : (
                      <div className="p-4 text-center text-gray-500">
                        {loading ? 'Loading recommendations...' : 'No recommendations available. Please refresh.'}
                      </div>
                    )}
                  </div>
                </div>
              </div>
              {/* Sentiment Analysis (now full width) and News Feed */}
              {renderNewsFeed()}
            </div>
          )}

          {activeTab === 'sentiment' && (
            <div className="container mx-auto px-4 py-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Sentiment Analysis Input */}
                <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                  <h2 className="text-xl font-bold mb-4">Analyze Stock Sentiment</h2>
                  <div className="flex space-x-4">
                    <input
                      type="text"
                      placeholder="Enter stock symbol..."
                      className={`flex-1 px-4 py-2 rounded-lg border ${
                        darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-300'
                      } focus:outline-none focus:ring-2 focus:ring-blue-500`}
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !loading && searchQuery.trim()) {
                          handleSentimentAnalysis(searchQuery);
                        }
                      }}
                    />
                    <button
                      onClick={() => handleSentimentAnalysis(searchQuery)}
                      className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                      disabled={loading || !searchQuery.trim()}
                    >
                      {loading ? 'Analyzing...' : 'Analyze'}
                    </button>
                  </div>
                  {sentimentData && (
                    <div className="mt-4">
                      <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                        Showing analysis results
                      </p>
                    </div>
                  )}
                </div>

                {/* Overall Sentiment Gauge */}
                <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                  <h2 className="text-xl font-bold mb-4">Overall Sentiment</h2>
                  <div id="overall-sentiment-gauge" className="w-full h-56 flex items-center justify-center"></div>
                  {sentimentData?.overall_sentiment && (
                    <div className="text-center mt-2">
                      <div className="text-lg font-bold" style={{ color: sentimentData.overall_sentiment.score > 0 ? '#10B981' : sentimentData.overall_sentiment.score < 0 ? '#EF4444' : '#6B7280' }}>
                        {sentimentData.overall_sentiment.label}
                      </div>
                    </div>
                  )}
                </div>

                {/* Explainable AI: why Bullish / Bearish / Neutral */}
                {sentimentData?.explanation && (
                  <div className={`lg:col-span-2 p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                    <h2 className="text-xl font-bold mb-2">Why this sentiment label?</h2>
                    <p className={`text-sm mb-3 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      {sentimentData.explanation.description}
                    </p>
                    {sentimentData.explanation.uses_demo_news && (
                      <p className={`text-sm mb-3 rounded-md px-3 py-2 ${darkMode ? 'bg-amber-900/40 text-amber-200' : 'bg-amber-50 text-amber-900'}`}>
                        Sample demo news is active; per-article scores are illustrative. Keyword matches still reflect the text.
                      </p>
                    )}
                    <div className={`p-4 rounded-lg border-l-4 mb-4 ${darkMode ? 'bg-gray-700 border-cyan-500' : 'bg-gray-50 border-cyan-500'}`}>
                      <p className={`text-sm leading-relaxed ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                        {sentimentData.explanation.label_rationale}
                      </p>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                      <div>
                        <h3 className={`text-sm font-semibold mb-2 ${darkMode ? 'text-emerald-300' : 'text-emerald-700'}`}>
                          Top positive lexicon terms
                        </h3>
                        {sentimentData.explanation.top_positive_terms.length > 0 ? (
                          <ul className={`text-sm space-y-1 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                            {sentimentData.explanation.top_positive_terms.slice(0, 10).map((row) => (
                              <li key={row.term}>
                                <span className="font-medium">{row.term}</span>
                                <span className={darkMode ? 'text-gray-500' : 'text-gray-500'}> — in {row.article_count} article(s)</span>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className={`text-sm ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>No positive lexicon hits in this batch.</p>
                        )}
                      </div>
                      <div>
                        <h3 className={`text-sm font-semibold mb-2 ${darkMode ? 'text-rose-300' : 'text-rose-700'}`}>
                          Top negative lexicon terms
                        </h3>
                        {sentimentData.explanation.top_negative_terms.length > 0 ? (
                          <ul className={`text-sm space-y-1 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                            {sentimentData.explanation.top_negative_terms.slice(0, 10).map((row) => (
                              <li key={row.term}>
                                <span className="font-medium">{row.term}</span>
                                <span className={darkMode ? 'text-gray-500' : 'text-gray-500'}> — in {row.article_count} article(s)</span>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className={`text-sm ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>No negative lexicon hits in this batch.</p>
                        )}
                      </div>
                    </div>
                    <details className={`rounded-lg border ${darkMode ? 'border-gray-600 bg-gray-900/40' : 'border-gray-200 bg-gray-50'}`}>
                      <summary className={`cursor-pointer px-4 py-3 text-sm font-medium ${darkMode ? 'text-gray-200' : 'text-gray-800'}`}>
                        Per-article explanations ({sentimentData.explanation.per_article.length})
                      </summary>
                      <div className={`max-h-80 overflow-y-auto px-4 pb-4 space-y-3 border-t ${darkMode ? 'border-gray-600' : 'border-gray-200'}`}>
                        {sentimentData.explanation.per_article.map((row, i) => (
                          <div
                            key={`${i}-${row.title.slice(0, 24)}`}
                            className={`pt-3 text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}
                          >
                            <div className={`font-medium ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>{row.title}</div>
                            <div className={`mt-1 text-xs ${darkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                              Score {row.sentiment.toFixed(2)} · source: {row.score_source} · weighted +{row.weighted_positive_hits} / −{row.weighted_negative_hits}
                            </div>
                            <p className="mt-2 leading-relaxed">{row.why}</p>
                          </div>
                        ))}
                      </div>
                    </details>
                  </div>
                )}

                {/* Sentiment Timeline Chart */}
                <div className={`lg:col-span-2 p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                  <h2 className="text-xl font-bold mb-4">Sentiment Timeline</h2>
                  <div className="w-full h-80 flex items-center justify-center">
                    {sentimentData?.sentiment_timeline && sentimentData.sentiment_timeline.length > 0 ? (
                      <div id="sentiment-chart" className="w-full h-80"></div>
                    ) : (
                      <div className="text-gray-500 text-center w-full">
                        No sentiment timeline data available. Analyze a stock to see sentiment trends.
                      </div>
                    )}
                  </div>
                </div>

                {/* 10-Year Price History Chart */}
                <div className={`lg:col-span-2 p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                  <h2 className="text-xl font-bold mb-4">1-Year Price History</h2>
                  <div className="w-full h-80 flex items-center justify-center">
                    {sentimentData?.price_history_10y && sentimentData.price_history_10y.length > 0 ? (
                      <div id="price-history-10y-chart" className="w-full h-80"></div>
                    ) : (
                      <div className="text-gray-500 text-center w-full">
                        No price history data available for this ticker. Please try another symbol or check back later.
                      </div>
                    )}
                  </div>
                </div>

                {/* Key Findings */}
                <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                  <h2 className="text-xl font-bold mb-4">Key Findings</h2>
                  <div className="space-y-3">
                    {sentimentData?.key_findings && sentimentData.key_findings.length > 0 ? (
                      sentimentData.key_findings.map((finding, index) => (
                      <div key={index} className={`p-4 rounded-lg border-l-4 ${
                        darkMode 
                          ? 'bg-gray-700 border-blue-500' 
                          : 'bg-gray-50 border-blue-400'
                      }`}>
                        <div className={`prose prose-sm max-w-none ${darkMode ? 'prose-invert' : ''}`}>
                          <ReactMarkdown
                            components={{
                              p: ({node, ...props}) => <p className={`mb-0 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                              ul: ({node, ...props}) => <ul className={`list-disc list-inside mb-0 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                              li: ({node, ...props}) => <li className={`${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                              strong: ({node, ...props}) => <strong className="font-semibold" {...props} />,
                            }}
                          >
                            {finding}
                          </ReactMarkdown>
                        </div>
                      </div>
                      ))
                    ) : (
                      <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700 text-gray-400' : 'bg-gray-50 text-gray-600'}`}>
                        No key findings available.
                      </div>
                    )}
                  </div>
                </div>

                {/* Recommendations */}
                <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                  <h2 className="text-xl font-bold mb-4">Recommendations</h2>
                  <div className="space-y-3">
                    {sentimentData?.recommendations && sentimentData.recommendations.length > 0 ? (
                      sentimentData.recommendations.map((recommendation, index) => (
                      <div key={index} className={`p-4 rounded-lg border-l-4 ${
                        darkMode 
                          ? 'bg-gray-700 border-green-500' 
                          : 'bg-gray-50 border-green-400'
                      }`}>
                        <div className={`prose prose-sm max-w-none ${darkMode ? 'prose-invert' : ''}`}>
                          <ReactMarkdown
                            components={{
                              p: ({node, ...props}) => <p className={`mb-0 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                              ul: ({node, ...props}) => <ul className={`list-disc list-inside mb-0 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                              li: ({node, ...props}) => <li className={`${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                              strong: ({node, ...props}) => <strong className="font-semibold" {...props} />,
                            }}
                          >
                            {recommendation}
                          </ReactMarkdown>
                        </div>
                      </div>
                      ))
                    ) : (
                      <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700 text-gray-400' : 'bg-gray-50 text-gray-600'}`}>
                        No recommendations available.
                      </div>
                    )}
                  </div>
                </div>

                {/* Top News */}
                <div className={`lg:col-span-2 p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                  <h2 className="text-xl font-bold mb-4">Top News</h2>
                  <div className="space-y-4">
                    {sentimentData?.top_news && sentimentData.top_news.length > 0 ? (
                      sentimentData.top_news.map((article, index) => (
                      <div key={index} className={`p-4 rounded-lg ${
                        darkMode ? 'bg-gray-700' : 'bg-gray-50'
                      }`}>
                        <div className="flex justify-between items-start">
                          <div>
                            <h3 className="font-bold">{article.title}</h3>
                            <p className="text-sm text-gray-500">{article.source} • {article.timestamp}</p>
                          </div>
                          <div className={`px-2 py-1 rounded text-xs ${
                            article.sentiment > 0
                              ? darkMode ? 'bg-green-900 text-green-400' : 'bg-green-100 text-green-800'
                              : article.sentiment < 0
                              ? darkMode ? 'bg-red-900 text-red-400' : 'bg-red-100 text-red-800'
                              : darkMode ? 'bg-gray-600 text-gray-300' : 'bg-gray-200 text-gray-700'
                          }`}>
                            {article.sentiment > 0 ? 'Bullish' : article.sentiment < 0 ? 'Bearish' : 'Neutral'}
                          </div>
                        </div>
                        {article.content && (
                          <p className="text-sm mt-2">{article.content}</p>
                        )}
                        {/* Display credibility stars */}
                        {article.credibility !== undefined && (
                          <div className="flex items-center mt-2 text-yellow-500">
                            {Array(5).fill(0).map((_, starIndex) => (
                              <i
                                key={starIndex}
                                className={`fas fa-star ${starIndex < article.credibility ? '' : 'text-gray-300'}`}
                              ></i>
                            ))}
                          </div>
                        )}
                        {/* Display Read Full Article link */}
                        {article.url && (
                          <div className="mt-2">
                            <a href={article.url} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline text-sm">
                              Read full article
                            </a>
                          </div>
                        )}
                      </div>
                      ))
                    ) : (
                      <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700 text-gray-400' : 'bg-gray-50 text-gray-600'}`}>
                        No news articles available for this ticker.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'fundamental' && (
            <div className="container mx-auto px-4 py-6">
              {/* Top Card: Heading + Search */}
              <div className="bg-white rounded-xl shadow p-8 mb-6 flex flex-col gap-4">
                <h1 className="text-2xl font-bold mb-2">Fundamental Analysis</h1>
                <div className="flex w-full">
                  <input
                    type="text"
                    placeholder="Enter stock symbol or company name..."
                    className="flex-1 px-4 py-3 rounded-l-lg border-0 bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 text-lg"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                  <button
                    onClick={() => handleFundamentalAnalysis(searchQuery)}
                    className="px-8 py-3 bg-blue-600 text-white rounded-r-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 text-lg font-semibold"
                    disabled={loading}
                  >
                    {loading ? 'Analyzing...' : 'Analyze'}
                  </button>
                </div>
              </div>
              {/* Row 1: Company Info & AI Insights */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-6">
                {/* Company Info Card */}
                <div className="bg-white rounded-xl shadow p-8 flex flex-col gap-6">
                  <div className="flex items-center gap-6 mb-4">
                    <div className="h-14 w-14 rounded-full bg-blue-100 flex items-center justify-center">
                      <i className="fas fa-building text-blue-600 text-3xl"></i>
                    </div>
                    <div>
                      <div className="text-2xl font-bold leading-tight">{fundamentalData?.company_info?.name || 'Company Name'}</div>
                      <div className="text-xs text-gray-500 font-semibold uppercase tracking-wide mt-1">
                        {fundamentalData?.company_info?.exchange ? `${fundamentalData.company_info.exchange}:` : ''} {fundamentalData?.company_info?.symbol || ''}
                      </div>
                    </div>
                  </div>
                  <table className="w-full text-base">
                    <tbody>
                      <tr className="border-b border-gray-200">
                        <td className="py-2 text-gray-600 font-medium">Current Price</td>
                        <td className="py-2 text-right font-bold">{fundamentalData?.key_metrics?.current_price !== undefined ? `$${fundamentalData.key_metrics.current_price}` : '—'}</td>
                      </tr>
                      <tr className="border-b border-gray-200">
                        <td className="py-2 text-gray-600 font-medium">Market Cap</td>
                        <td className="py-2 text-right font-bold">{fundamentalData?.key_metrics?.market_cap !== undefined ? `$${Number(fundamentalData.key_metrics.market_cap).toLocaleString()}` : '—'}</td>
                      </tr>
                      <tr className="border-b border-gray-200">
                        <td className="py-2 text-gray-600 font-medium">P/E Ratio</td>
                        <td className="py-2 text-right font-bold">{fundamentalData?.key_metrics?.pe_ratio !== undefined ? fundamentalData.key_metrics.pe_ratio : '—'}</td>
                      </tr>
                      <tr className="border-b border-gray-200">
                        <td className="py-2 text-gray-600 font-medium">EPS (TTM)</td>
                        <td className="py-2 text-right font-bold">{fundamentalData?.key_metrics?.eps_ttm !== undefined ? fundamentalData.key_metrics.eps_ttm : '—'}</td>
                      </tr>
                      <tr className="border-b border-gray-200">
                        <td className="py-2 text-gray-600 font-medium">Dividend Yield</td>
                        <td className="py-2 text-right font-bold">{fundamentalData?.key_metrics?.dividend_yield !== undefined ? `${fundamentalData.key_metrics.dividend_yield}%` : '—'}</td>
                      </tr>
                      <tr className="border-b border-gray-200">
                        <td className="py-2 text-gray-600 font-medium">52-Week Range</td>
                        <td className="py-2 text-right font-bold">{fundamentalData?.key_metrics?.['52_week_low'] !== undefined && fundamentalData?.key_metrics?.['52_week_high'] !== undefined ? `$${fundamentalData.key_metrics['52_week_low']} - $${fundamentalData.key_metrics['52_week_high']}` : '—'}</td>
                      </tr>
                      <tr>
                        <td className="py-2 text-gray-600 font-medium">Beta</td>
                        <td className="py-2 text-right font-bold">{fundamentalData?.key_metrics?.beta !== undefined ? fundamentalData.key_metrics.beta : '—'}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                {/* AI Insights Card */}
                <div className="bg-white rounded-xl shadow p-8 flex flex-col gap-4">
                  <h2 className="text-xl font-bold mb-2">AI-Generated Financial Insights</h2>
                  <div className="space-y-3">
                    {fundamentalData?.ai_insights && Object.entries(fundamentalData.ai_insights)
                      .filter(([key]) => [
                        'revenue_growth',
                        'profitability',
                        'balance_sheet_strength',
                        'valuation_assessment'
                      ].includes(key))
                      .map(([key, value]) => (
                        <div key={key} className="p-4 rounded-lg bg-gray-50">
                          <div className="font-semibold mb-1">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
                          <div className="text-base text-gray-700">{value as string}</div>
                        </div>
                    ))}
                  </div>
                </div>
              </div>
              {/* Row 2: Financial Ratios & Competitive Analysis */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-6">
                {/* Financial Ratios Card */}
                <div className="bg-white rounded-xl shadow p-8">
                  <h2 className="text-xl font-bold mb-4">Financial Ratios</h2>
                  <div className="space-y-4">
                    {fundamentalData?.financial_ratios && fundamentalData.financial_ratios.map((ratio: any, idx: number) => (
                      <div key={ratio.metric} className="flex items-center">
                        <span className="w-40 font-semibold text-base">{ratio.metric}</span>
                        <div className="flex-1 mx-4 h-4 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className={`h-4 rounded-full ${['bg-green-500','bg-blue-500','bg-yellow-500','bg-purple-500','bg-pink-500'][idx % 5]}`}
                            style={{ width: `${Math.min(100, Math.max(0, ratio.score))}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                {/* Competitive Analysis Card */}
                <div className="bg-white rounded-xl shadow p-8">
                  <h2 className="text-xl font-bold mb-4">Competitive Analysis</h2>
                  <div className="overflow-x-auto">
                    {fundamentalData?.competitive_analysis && fundamentalData.competitive_analysis.length > 0 && (
                      <table className="min-w-full text-base">
                        <thead>
                          <tr>
                            <th className="px-2 py-1 text-left font-semibold">Metric</th>
                            {fundamentalData.competitive_analysis.map((comp: any) => (
                              <th key={comp.symbol} className="px-2 py-1 text-center font-semibold">{comp.symbol}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {['pe_ratio','revenue_growth','gross_margin','roe','dividend_yield'].map((metric: string) => (
                            <tr key={metric}>
                              <td className="px-2 py-1 capitalize">{metric.replace(/_/g,' ').replace('roe','ROE')}</td>
                              {fundamentalData.competitive_analysis.map((comp: any) => (
                                <td key={comp.symbol+metric} className="px-2 py-1 text-center">{comp[metric]}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>
                </div>
              </div>
              {/* Row 3: Financial Statement Analysis (full width) */}
              <div className="bg-white rounded-xl shadow p-8 mb-6">
                <h2 className="text-xl font-bold mb-4">Financial Statement Analysis</h2>
                <div className="mb-4">
                  <div className="flex space-x-2 mb-4">
                    <button
                      className={`px-4 py-2 text-base rounded-full cursor-pointer !rounded-button ${statementTab === 'income_statement' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
                      onClick={() => setStatementTab('income_statement')}
                    >
                      Income Statement
                    </button>
                    <button
                      className={`px-4 py-2 text-base rounded-full cursor-pointer !rounded-button ${statementTab === 'balance_sheet' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
                      onClick={() => setStatementTab('balance_sheet')}
                    >
                      Balance Sheet
                    </button>
                    <button
                      className={`px-4 py-2 text-base rounded-full cursor-pointer !rounded-button ${statementTab === 'cash_flow' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-700'}`}
                      onClick={() => setStatementTab('cash_flow')}
                    >
                      Cash Flow
                    </button>
                  </div>
                  <div className="mt-2 overflow-x-auto">
                    {(() => {
                      const data = fundamentalData?.financial_statements?.[statementTab] || [];
                      if (!Array.isArray(data) || data.length === 0) {
                        return <div className="text-gray-500">No data available</div>;
                      }
                      // Get all unique metric names (row headers) except 'period'
                      const metricNames = Object.keys(data[0]).filter((k) => k !== 'period');
                      // Get all periods (column headers)
                      const periods = data.map((row) => row.period);
                      return (
                        <table className="min-w-full text-base">
                          <thead>
                            <tr>
                              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Metric ($ Millions)</th>
                              {periods.map((period, idx) => (
                                <th key={period} className="px-4 py-2 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">{period}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {metricNames.map((metric) => (
                              <tr key={metric} className="border-b border-gray-200">
                                <td className="py-2 text-base font-medium">{metric}</td>
                                {data.map((row, idx) => (
                                  <td key={idx} className="py-2 text-base text-center">{row[metric]}</td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      );
                    })()}
                  </div>
                </div>
                {/* AI Financial Analysis (summary) */}
                {fundamentalData?.ai_insights?.financial_analysis && (
                  <div className={`mt-6 p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                    <h3 className={`text-xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>AI Financial Analysis</h3>
                    <div className={`prose max-w-none ${darkMode ? 'prose-invert' : ''} prose-sm`}>
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          h1: ({node, ...props}) => <h1 className={`text-2xl font-bold mt-6 mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`} {...props} />,
                          h2: ({node, ...props}) => <h2 className={`text-xl font-bold mt-5 mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`} {...props} />,
                          h3: ({node, ...props}) => <h3 className={`text-lg font-semibold mt-4 mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`} {...props} />,
                          h4: ({node, ...props}) => <h4 className={`text-md font-semibold mt-3 mb-1 ${darkMode ? 'text-white' : 'text-gray-900'}`} {...props} />,
                          p: ({node, ...props}) => <p className={`mb-3 leading-relaxed ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                          ul: ({node, ...props}) => <ul className={`list-disc list-inside mb-4 space-y-1 ml-4 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                          ol: ({node, ...props}) => <ol className={`list-decimal list-inside mb-4 space-y-1 ml-4 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                          li: ({node, ...props}) => <li className={`mb-1 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                          strong: ({node, ...props}) => <strong className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`} {...props} />,
                          code: ({node, inline, ...props}: any) => inline ? (
                            <code className={`px-1.5 py-0.5 rounded text-sm ${darkMode ? 'bg-gray-700 text-gray-200' : 'bg-gray-100 text-gray-800'}`} {...props} />
                          ) : (
                            <code className={`block p-3 rounded-md text-sm overflow-x-auto ${darkMode ? 'bg-gray-900 text-gray-200' : 'bg-gray-100 text-gray-800'}`} {...props} />
                          ),
                          blockquote: ({node, ...props}) => <blockquote className={`border-l-4 pl-4 italic my-4 ${darkMode ? 'border-gray-500 text-gray-400 bg-gray-700/30' : 'border-gray-300 text-gray-600 bg-gray-50'}`} {...props} />,
                          table: ({node, ...props}: any) => (
                            <div className="overflow-x-auto my-6">
                              <table className={`min-w-full border-collapse ${darkMode ? 'border-gray-600' : 'border-gray-300'} border rounded-lg`} {...props} />
                            </div>
                          ),
                          thead: ({node, ...props}: any) => <thead className={`${darkMode ? 'bg-gray-700' : 'bg-gray-100'} border-b ${darkMode ? 'border-gray-600' : 'border-gray-300'}`} {...props} />,
                          tbody: ({node, ...props}: any) => <tbody className={darkMode ? 'divide-y divide-gray-600' : 'divide-y divide-gray-200'} {...props} />,
                          tr: ({node, ...props}: any) => <tr className={`border-b ${darkMode ? 'border-gray-600 hover:bg-gray-700/30' : 'border-gray-200 hover:bg-gray-50'} transition-colors`} {...props} />,
                          th: ({node, ...props}: any) => <th className={`px-4 py-3 text-left text-sm font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`} {...props} />,
                          td: ({node, ...props}: any) => <td className={`px-4 py-3 text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                          hr: ({node, ...props}: any) => <hr className={`my-4 ${darkMode ? 'border-gray-600' : 'border-gray-300'}`} {...props} />,
                        }}
                      >
                        {fundamentalData.ai_insights.financial_analysis}
                      </ReactMarkdown>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'portfolio' && (
            <div className="container mx-auto px-4 py-6">
              <div className={`p-6 rounded-lg shadow-md mb-6 ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <h2 className="text-xl font-bold mb-4">Portfolio Optimization</h2>
                <div className="mb-6">
                  <div className={`border-2 border-dashed ${darkMode ? 'border-gray-600' : 'border-gray-300'} rounded-lg p-8 text-center`}>
                    <div className="flex flex-col items-center">
                      <i className="fas fa-cloud-upload-alt text-4xl mb-3 text-blue-500"></i>
                      <h3 className="text-lg font-medium mb-2">Upload Portfolio</h3>
                      <p className="text-sm text-gray-500 mb-4">Upload a CSV file of your current portfolio.<br />
                        <span className='block mt-2 text-xs text-gray-400'>Format: <b>ticker, amount, price</b> (last row: <b>TOTAL_WORTH,,[total]</b>)</span>
                        <span className='block mt-1 text-xs text-gray-400'>Example:<br />AAPL,10,180<br />MSFT,5,320<br />GOOGL,2,2700<br />TOTAL_WORTH,,10000</span>
                      </p>
                      <input
                        type="file"
                        accept=".csv"
                        onChange={handleFileUpload}
                        className="hidden"
                        id="portfolio-upload"
                      />
                      <label
                        htmlFor="portfolio-upload"
                        className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm cursor-pointer hover:bg-blue-700 transition-colors"
                      >
                        Choose File
                      </label>
                    </div>
                  </div>
                </div>

                {!portfolioData && (
                  <div className="text-center py-8">
                    <p className="text-gray-500">Upload your portfolio to get started with optimization</p>
                  </div>
                )}

                {portfolioData && (
                  <>
                    {/* Risk Profile Selection */}
                    <div className={`p-6 rounded-lg shadow-md mb-6 ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                      <h2 className="text-xl font-bold mb-4">Risk Profile Selection</h2>
                      <div className="mb-6">
                        <h3 className="text-lg font-semibold mb-3">
                          Current Risk Profile: <span className="text-yellow-500">{portfolioData.current_portfolio.risk_level || 'N/A'}</span>
                        </h3>
                        <p className="text-sm mb-4">Select your risk tolerance to optimize your portfolio:</p>
                        <div className="grid grid-cols-3 gap-4 mb-4">
                          <button
                            onClick={() => handleRiskProfileChange('Conservative')}
                            disabled={!uploadedFile}
                            className={`p-4 rounded-lg border-2 transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500
                              ${selectedRiskProfile === 'Conservative'
                                ? darkMode
                                  ? 'border-blue-400 bg-blue-900 text-blue-200'
                                  : 'border-blue-600 bg-blue-100 text-blue-800'
                                : darkMode
                                  ? 'border-gray-700 bg-gray-900 text-gray-300 hover:border-blue-500 hover:bg-blue-950'
                                  : 'border-gray-200 bg-white text-gray-700 hover:border-blue-400 hover:bg-blue-50'}
                              ${!uploadedFile ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                            tabIndex={0}
                            aria-pressed={selectedRiskProfile === 'Conservative'}
                          >
                            <h4 className="font-semibold">Conservative</h4>
                            <p className="text-sm">Lower risk, stable returns</p>
                          </button>
                          <button
                            onClick={() => handleRiskProfileChange('Moderate')}
                            disabled={!uploadedFile}
                            className={`p-4 rounded-lg border-2 transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-yellow-500
                              ${selectedRiskProfile === 'Moderate'
                                ? darkMode
                                  ? 'border-yellow-400 bg-yellow-900 text-yellow-200'
                                  : 'border-yellow-600 bg-yellow-100 text-yellow-800'
                                : darkMode
                                  ? 'border-gray-700 bg-gray-900 text-gray-300 hover:border-yellow-500 hover:bg-yellow-950'
                                  : 'border-gray-200 bg-white text-gray-700 hover:border-yellow-400 hover:bg-yellow-50'}
                              ${!uploadedFile ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                            tabIndex={0}
                            aria-pressed={selectedRiskProfile === 'Moderate'}
                          >
                            <h4 className="font-semibold">Moderate</h4>
                            <p className="text-sm">Balanced risk and returns</p>
                          </button>
                          <button
                            onClick={() => handleRiskProfileChange('Aggressive')}
                            disabled={!uploadedFile}
                            className={`p-4 rounded-lg border-2 transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-red-500
                              ${selectedRiskProfile === 'Aggressive'
                                ? darkMode
                                  ? 'border-red-400 bg-red-900 text-red-200'
                                  : 'border-red-600 bg-red-100 text-red-800'
                                : darkMode
                                  ? 'border-gray-700 bg-gray-900 text-gray-300 hover:border-red-500 hover:bg-red-950'
                                  : 'border-gray-200 bg-white text-gray-700 hover:border-red-400 hover:bg-red-50'}
                              ${!uploadedFile ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                            tabIndex={0}
                            aria-pressed={selectedRiskProfile === 'Aggressive'}
                          >
                            <h4 className="font-semibold">Aggressive</h4>
                            <p className="text-sm">Higher risk, higher returns</p>
                          </button>
                        </div>
                      </div>
                    </div>

                    {/* Current Portfolio & AI-Recommended Optimization */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                      {/* Current Portfolio */}
                      <div>
                        <h3 className="text-lg font-semibold mb-3">Current Portfolio</h3>
                        <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>                
                          <div className="mb-4">
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-sm">Total Value</span>
                              <span className="font-bold">${portfolioData.current_portfolio.total_value.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-sm">Annualized Return</span>
                              <span className="font-medium text-green-500">
                                {portfolioData.current_portfolio.annualized_return > 0 ? '+' : ''}
                                {portfolioData.current_portfolio.annualized_return.toFixed(1)}%
                              </span>
                            </div>
                            <div className="flex justify-between items-center">
                              <span className="text-sm">Risk Level</span>
                              <span className="font-medium text-yellow-500">{portfolioData.current_portfolio.risk_level}</span>
                            </div>
                          </div>
                          <div className="mb-4">
                            <h4 className="font-semibold mb-2">Asset Allocation</h4>
                            <div className="flex flex-wrap gap-2">
                              {portfolioData.current_portfolio.asset_allocation.map((a) => (
                                <span key={a.category} className="px-2 py-1 rounded bg-blue-100 text-blue-800 text-xs">
                                  {a.category} ({a.percentage}%)
                                </span>
                              ))}
                            </div>
                          </div>
                          <div>
                            <h4 className="font-semibold mb-2">Top Holdings</h4>
                            <ul className="text-sm">
                              {portfolioData.current_portfolio.top_holdings.map((h) => (
                                <li key={h.ticker} className="flex justify-between items-center mb-2">
                                  <div>
                                    <span className="font-bold">{h.company_name}</span> <span className="text-xs text-gray-500">({h.ticker})</span>
                                  </div>
                                  <div className="text-right">
                                    <span className="text-xs">{h.amount} shares</span><br />
                                    <span className="text-xs text-gray-500">{h.percentage}% of portfolio</span>
                                  </div>
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                      {/* AI-Recommended Optimization */}
                      <div>
                        <h3 className="text-lg font-semibold mb-3">Optimized Portfolio</h3>
                        <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>                
                          <div className="mb-4">
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-sm">Projected Value (1Y)</span>
                              <span className="font-bold">${portfolioData.recommended_optimization.projected_value.toLocaleString()}</span>
                            </div>
                            <div className="flex justify-between items-center mb-2">
                              <span className="text-sm">Projected Return</span>
                              <span className="font-medium text-green-500">
                                {portfolioData.recommended_optimization.projected_return > 0 ? '+' : ''}
                                {portfolioData.recommended_optimization.projected_return.toFixed(1)}%
                              </span>
                            </div>
                            <div className="flex justify-between items-center">
                              <span className="text-sm">Optimized Risk Level</span>
                              <span className="font-medium text-blue-500">{portfolioData.recommended_optimization.optimized_risk_level}</span>
                            </div>
                          </div>
                          <div className="mb-4">
                            <h4 className="font-semibold mb-2">Optimized Allocation</h4>
                            <div className="overflow-x-auto">
                              <table className="min-w-full text-sm">
                                <thead>
                                  <tr>
                                    <th className="px-2 py-1 text-left font-semibold">Ticker</th>
                                    <th className="px-2 py-1 text-left font-semibold">Company Name</th>
                                    <th className="px-2 py-1 text-center font-semibold">Allocation %</th>
                                    <th className="px-2 py-1 text-center font-semibold">Shares</th>
                                    <th className="px-2 py-1 text-center font-semibold">Price</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {portfolioData.recommended_optimization.optimized_allocation.map((h: any) => (
                                    <tr key={h.ticker}>
                                      <td className="px-2 py-1 font-bold">{h.ticker}</td>
                                      <td className="px-2 py-1">{h.company_name || '-'}</td>
                                      <td className="px-2 py-1 text-center">{h.optimal_percentage != null ? h.optimal_percentage + '%' : '-'}</td>
                                      <td className="px-2 py-1 text-center">{h.shares != null ? h.shares : '-'}</td>
                                      <td className="px-2 py-1 text-center">{h.price != null ? `$${Number(h.price).toFixed(2)}` : '-'}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                          <div className="mb-4">
                            <h4 className="font-semibold mb-2">Recommended Actions</h4>
                            {portfolioData.recommended_optimization.recommended_actions && portfolioData.recommended_optimization.recommended_actions.length > 0 ? (
                              <ul className="text-sm">
                                {portfolioData.recommended_optimization.recommended_actions.map((a: any, idx: number) => (
                                  <li key={idx} className="mb-1">
                                    <span className="font-bold">{a.action} {a.company_name} ({a.ticker}):</span> {a.details}
                                  </li>
                                ))}
                                {/* Add AI-suggested tickers as actions */}
                                {portfolioData.ai_analysis && (() => {
                                  const existingTickers = portfolioData.recommended_optimization.optimized_allocation.map((h: any) => h.ticker);
                                  const suggested = extractSuggestedTickers(portfolioData.ai_analysis, existingTickers);
                                  return suggested.map((ticker, idx) => (
                                    <li key={`ai-suggested-${ticker}`} className="mb-1">
                                      <span className="font-bold">Add {ticker}:</span> Consider adding {ticker} to your portfolio.
                                    </li>
                                  ));
                                })()}
                              </ul>
                            ) : (
                              <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                                {portfolioData.ai_analysis 
                                  ? "No specific actions recommended at this time. Review the AI analysis below for general guidance."
                                  : "No actions available. AI analysis is being generated..."}
                              </p>
                            )}
                          </div>
                          {portfolioData.recommended_optimization.tax_loss_harvesting && portfolioData.recommended_optimization.tax_loss_harvesting.length > 0 && (
                            <div>
                              <h4 className="font-semibold mb-2">Tax-Loss Harvesting Opportunities</h4>
                              <ul className="text-sm">
                                {portfolioData.recommended_optimization.tax_loss_harvesting.map((t, idx) => (
                                  <li key={idx}>{t}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* AI Suggested Buys (parsed from ai_analysis) */}
                    {portfolioData.ai_analysis && (() => {
                      const existingTickers = portfolioData.recommended_optimization.optimized_allocation.map((h: any) => h.ticker);
                      const suggested = extractSuggestedTickers(portfolioData.ai_analysis, existingTickers);
                      if (suggested.length === 0) return null;
                      return (
                        <div className={`p-6 rounded-lg shadow-md mb-6 ${darkMode ? 'bg-gray-800 text-white' : 'bg-white text-black'}`}>
                          <h2 className="text-xl font-bold mb-4">AI Suggested Buys</h2>
                          <ul className="text-sm flex flex-wrap gap-2 mb-2">
                            {suggested.map(ticker => (
                              <li key={ticker} className="px-2 py-1 rounded bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">{ticker}</li>
                            ))}
                          </ul>
                          <p className="text-xs text-gray-500 mt-1">Consider adding these stocks to further optimize your portfolio as per AI recommendations.</p>
                        </div>
                      );
                    })()}

                    {/* AI-Driven Stock Suggestions (structured from backend) */}
                    {Array.isArray(portfolioData.ai_suggestions) && portfolioData.ai_suggestions.length > 0 && (
                      <div className={`p-6 rounded-lg shadow-md mb-6 ${darkMode ? 'bg-gray-800 text-white' : 'bg-white text-black'}`}>
                        <h2 className="text-xl font-bold mb-4">AI-Driven Stock Suggestions</h2>
                        <div className="space-y-4">
                          {portfolioData.ai_suggestions.map((rec) => (
                            <div key={rec.ticker} className={`p-4 rounded-lg border-l-4 ${rec.sentiment > 0 ? (darkMode ? 'bg-green-900 bg-opacity-20 border-green-500' : 'bg-green-50 border-green-500') : rec.sentiment < 0 ? (darkMode ? 'bg-red-900 bg-opacity-20 border-red-500' : 'bg-red-50 border-red-500') : (darkMode ? 'bg-gray-700 bg-opacity-20 border-gray-400' : 'bg-gray-100 border-gray-400')}`}>
                              <div className="flex justify-between items-start">
                                <div>
                                  <h3 className="font-bold">{rec.ticker}</h3>
                                  <p className="text-sm text-gray-500">{rec.company_name}</p>
                                </div>
                                <div className="flex items-center">
                                  <span className={`font-bold mr-2 ${rec.sentiment > 0 ? 'text-green-500' : rec.sentiment < 0 ? 'text-red-500' : 'text-gray-500'}`}>{rec.sentiment > 0 ? `+${rec.sentiment.toFixed(2)}` : rec.sentiment.toFixed(2)}</span>
                                  {rec.sentiment > 0 ? (
                                    <i className="fas fa-arrow-up text-green-500"></i>
                                  ) : rec.sentiment < 0 ? (
                                    <i className="fas fa-arrow-down text-red-500"></i>
                                  ) : null}
                                </div>
                              </div>
                              <div className={`mt-2 prose prose-sm max-w-none ${darkMode ? 'prose-invert' : ''}`}>
                                <ReactMarkdown
                                  components={{
                                    p: ({node, ...props}) => <p className={`text-sm mb-0 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                                    strong: ({node, ...props}) => <strong className="font-semibold" {...props} />,
                                  }}
                                >
                                  {rec.summary}
                                </ReactMarkdown>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* AI Analysis and Suggestions */}
                    {(portfolioData.ai_analysis || (Array.isArray(portfolioData.ai_suggestions) && portfolioData.ai_suggestions.length > 0)) ? (
                      <div className={`p-6 rounded-lg shadow-md mb-6 ${darkMode ? 'bg-gray-800 text-white' : 'bg-white text-black'}`}>
                        <h2 className="text-xl font-bold mb-4">AI Portfolio Analysis</h2>
                        {portfolioData.ai_analysis && (
                          <div className="mb-6">
                            <div className={`prose max-w-none ${darkMode ? 'prose-invert' : ''} prose-sm prose-table:prose-sm`}>
                              <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                components={{
                                  h1: ({node, ...props}) => <h1 className={`text-2xl font-bold mt-6 mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`} {...props} />,
                                  h2: ({node, ...props}) => <h2 className={`text-xl font-bold mt-5 mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`} {...props} />,
                                  h3: ({node, ...props}) => <h3 className={`text-lg font-semibold mt-4 mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`} {...props} />,
                                  h4: ({node, ...props}) => <h4 className={`text-md font-semibold mt-3 mb-1 ${darkMode ? 'text-white' : 'text-gray-900'}`} {...props} />,
                                  p: ({node, ...props}) => <p className={`mb-3 leading-relaxed ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                                  ul: ({node, ...props}) => <ul className={`list-disc list-inside mb-4 space-y-1 ml-4 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                                  ol: ({node, ...props}) => <ol className={`list-decimal list-inside mb-4 space-y-1 ml-4 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                                  li: ({node, ...props}) => <li className={`mb-1 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                                  strong: ({node, ...props}) => <strong className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`} {...props} />,
                                  code: ({node, inline, ...props}: any) => inline ? (
                                    <code className={`px-1.5 py-0.5 rounded text-sm ${darkMode ? 'bg-gray-700 text-gray-200' : 'bg-gray-100 text-gray-800'}`} {...props} />
                                  ) : (
                                    <code className={`block p-3 rounded-md text-sm overflow-x-auto ${darkMode ? 'bg-gray-900 text-gray-200' : 'bg-gray-100 text-gray-800'}`} {...props} />
                                  ),
                                  blockquote: ({node, ...props}) => <blockquote className={`border-l-4 pl-4 italic my-4 ${darkMode ? 'border-gray-500 text-gray-400 bg-gray-700/30' : 'border-gray-300 text-gray-600 bg-gray-50'}`} {...props} />,
                                  table: ({node, ...props}: any) => (
                                    <div className="overflow-x-auto my-6">
                                      <table className={`min-w-full border-collapse ${darkMode ? 'border-gray-600' : 'border-gray-300'} border rounded-lg`} {...props} />
                                    </div>
                                  ),
                                  thead: ({node, ...props}: any) => <thead className={`${darkMode ? 'bg-gray-700' : 'bg-gray-100'} border-b ${darkMode ? 'border-gray-600' : 'border-gray-300'}`} {...props} />,
                                  tbody: ({node, ...props}: any) => <tbody className={darkMode ? 'divide-y divide-gray-600' : 'divide-y divide-gray-200'} {...props} />,
                                  tr: ({node, ...props}: any) => <tr className={`border-b ${darkMode ? 'border-gray-600 hover:bg-gray-700/30' : 'border-gray-200 hover:bg-gray-50'} transition-colors`} {...props} />,
                                  th: ({node, ...props}: any) => <th className={`px-4 py-3 text-left text-sm font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`} {...props} />,
                                  td: ({node, ...props}: any) => <td className={`px-4 py-3 text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`} {...props} />,
                                  hr: ({node, ...props}: any) => <hr className={`my-4 ${darkMode ? 'border-gray-600' : 'border-gray-300'}`} {...props} />,
                                }}
                              >
                                {portfolioData.ai_analysis}
                              </ReactMarkdown>
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className={`p-6 rounded-lg shadow-md mb-6 ${darkMode ? 'bg-gray-800 text-white' : 'bg-white text-black'}`}>
                        <h2 className="text-xl font-bold mb-4">AI Portfolio Analysis</h2>
                        <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                          AI analysis is being generated. Please refresh the page in a moment or check the backend logs for any errors.
                        </p>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default App; 