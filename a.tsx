// The exported code uses Tailwind CSS. Install Tailwind CSS in your dev environment to ensure all styles work.
import React, { useState, useEffect, useCallback } from 'react';
import * as echarts from 'echarts';
import axios from 'axios';

// API endpoints
const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000/api';

// Types
interface PortfolioData {
  totalValue: number;
  riskScore: number;
  sharpeRatio: number;
  overallSentiment: string;
  performance: {
    dates: string[];
    portfolioValues: number[];
    benchmarkValues: number[];
  };
  recommendations: Array<{
    symbol: string;
    name: string;
    sentiment: number;
    description: string;
  }>;
  sectorAllocation: Array<{
    name: string;
    value: number;
  }>;
}

interface NewsArticle {
  title: string;
  source: string;
  date: string;
  sentiment: number;
  label: string;
  confidence: number;
  reasons: string[];
  summary: string;
  credibility: number;
}

interface StockData {
  symbol: string;
  name: string;
  currentPrice: number;
  marketCap: number;
  peRatio: number;
  eps: number;
  dividendYield: number;
  weekRange: string;
  beta: number;
}

const App: React.FC = () => {
  // Existing state
  const [darkMode, setDarkMode] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [showNotifications, setShowNotifications] = useState<boolean>(false);
  const [showProfileMenu, setShowProfileMenu] = useState<boolean>(false);
  const [showPreview, setShowPreview] = useState<boolean>(false);
  const [uploadedFile, setUploadedFile] = useState<{
    name: string;
    size: string;
    type: string;
    content: string | ArrayBuffer | null;
  } | null>(null);

  // New state for real-time data
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [newsData, setNewsData] = useState<NewsArticle[]>([]);
  const [stockData, setStockData] = useState<StockData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch portfolio data
  const fetchPortfolioData = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/portfolio`);
      setPortfolioData(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch portfolio data');
      console.error('Error fetching portfolio data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch news data
  const fetchNewsData = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/news`);
      setNewsData(response.data);
    } catch (err) {
      console.error('Error fetching news data:', err);
    }
  }, []);

  // Fetch stock data
  const fetchStockData = useCallback(async (symbol: string) => {
    try {
      setLoading(true);
      const response = await axios.get(`${API_BASE_URL}/stock/${symbol}`);
      setStockData(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch stock data');
      console.error('Error fetching stock data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial data fetch
  useEffect(() => {
    fetchPortfolioData();
    fetchNewsData();
  }, [fetchPortfolioData, fetchNewsData]);

  // Real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      fetchPortfolioData();
      fetchNewsData();
    }, 60000); // Update every minute

    return () => clearInterval(interval);
  }, [fetchPortfolioData, fetchNewsData]);

  const processPortfolioFile = (file: typeof uploadedFile) => {
    if (!file) return;
    // Here you would implement the logic to process the uploaded file
    // and update the portfolio data accordingly
    console.log('Processing file:', file.name);
    // Show success message
    const successMessage = document.createElement('div');
    successMessage.className = `fixed bottom-4 right-4 px-6 py-3 rounded-lg ${darkMode ? 'bg-green-900' : 'bg-green-100'} text-green-700 z-50`;
    successMessage.textContent = 'Portfolio file uploaded successfully!';
    document.body.appendChild(successMessage);
    // Remove success message after 3 seconds
    setTimeout(() => {
      document.body.removeChild(successMessage);
    }, 3000);
  };

  useEffect(() => {
    // Initialize charts after component mounts
    const portfolioChartElement = document.getElementById('portfolio-chart');
    const sentimentChartElement = document.getElementById('sentiment-chart');
    const sectorAllocationElement = document.getElementById('sector-allocation');
    if (portfolioChartElement) {
      const portfolioChart = echarts.init(portfolioChartElement);
      const portfolioOption = {
        animation: false,
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'cross'
          }
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        },
        yAxis: {
          type: 'value'
        },
        series: [
          {
            name: 'Portfolio Value',
            type: 'line',
            smooth: true,
            lineStyle: {
              width: 3,
              color: '#3B82F6'
            },
            areaStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: 'rgba(59, 130, 246, 0.5)' },
                { offset: 1, color: 'rgba(59, 130, 246, 0.1)' }
              ])
            },
            data: [25000, 26200, 27800, 27100, 28500, 29800, 30500, 31200, 32500, 33100, 34200, 35500]
          },
          {
            name: 'Benchmark',
            type: 'line',
            smooth: true,
            lineStyle: {
              width: 2,
              color: '#9CA3AF'
            },
            data: [25000, 25600, 26200, 26800, 27500, 28100, 28700, 29300, 30000, 30600, 31200, 32000]
          }
        ]
      };
      portfolioChart.setOption(portfolioOption);
      window.addEventListener('resize', () => {
        portfolioChart.resize();
      });
    }
    if (sentimentChartElement) {
      const sentimentChart = echarts.init(sentimentChartElement);
      const sentimentOption = {
        animation: false,
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'shadow'
          }
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          data: ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA']
        },
        yAxis: {
          type: 'value',
          min: -1,
          max: 1
        },
        series: [
          {
            name: 'Sentiment Score',
            type: 'bar',
            data: [0.78, 0.65, -0.22, 0.45, 0.12, -0.35, 0.89],
            itemStyle: {
              color: function(params: any) {
                return params.data > 0 ? '#10B981' : '#EF4444';
              }
            }
          }
        ]
      };
      sentimentChart.setOption(sentimentOption);
      window.addEventListener('resize', () => {
        sentimentChart.resize();
      });
    }
    if (sectorAllocationElement) {
      const sectorChart = echarts.init(sectorAllocationElement);
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
          data: ['Technology', 'Healthcare', 'Consumer Cyclical', 'Financial Services', 'Communication Services', 'Others']
        },
        series: [
          {
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
            data: [
              { value: 35, name: 'Technology' },
              { value: 20, name: 'Healthcare' },
              { value: 15, name: 'Consumer Cyclical' },
              { value: 12, name: 'Financial Services' },
              { value: 10, name: 'Communication Services' },
              { value: 8, name: 'Others' }
            ]
          }
        ]
      };
      sectorChart.setOption(sectorOption);
      window.addEventListener('resize', () => {
        sectorChart.resize();
      });
    }
    return () => {
      window.removeEventListener('resize', () => {});
    };
  }, []);

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

  return (
    <div className={`min-h-screen ${darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'}`}>
      {/* Header */}
      <header className={`fixed w-full z-10 ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-md`}>
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center">
            <div className="text-2xl font-bold text-blue-600">
              <i className="fas fa-chart-line mr-2"></i>
              SentiInvest
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
                  <a href="https://readdy.ai/home/819de3fd-e4aa-4914-903c-a03fbf488bd6/3d1bc1ff-5cec-4b23-bb88-7e49b28c8c1c" data-readdy="true" className="block px-4 py-2 text-sm hover:bg-gray-100 hover:bg-opacity-20">Profile</a>
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
      <main className="pt-16 pb-16 md:pb-0 min-h-screen">
        {activeTab === 'dashboard' && (
          <div className="container mx-auto px-4 py-6">
            {/* Metrics Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <div className="flex justify-between items-start">
                  <div>
                    <p className="text-sm text-gray-500">Portfolio Value</p>
                    <h3 className="text-2xl font-bold">$35,500.00</h3>
                    <p className="text-sm text-green-500 flex items-center mt-1">
                      <i className="fas fa-arrow-up mr-1"></i>
                      3.8% ($1,300)
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
                    <h3 className="text-2xl font-bold">62/100</h3>
                    <p className="text-sm text-yellow-500 flex items-center mt-1">
                      <i className="fas fa-exclamation-triangle mr-1"></i>
                      Moderate
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
                    <h3 className="text-2xl font-bold">1.42</h3>
                    <p className="text-sm text-green-500 flex items-center mt-1">
                      <i className="fas fa-arrow-up mr-1"></i>
                      0.08
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
                    <h3 className="text-2xl font-bold">Bullish</h3>
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
            {/* Portfolio Chart and Sentiment Recommendations */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
              <div className={`lg:col-span-2 p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-bold">Portfolio Performance</h2>
                  <div className="flex space-x-2">
                    <button className={`px-3 py-1 text-sm rounded-full cursor-pointer !rounded-button ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>1M</button>
                    <button className={`px-3 py-1 text-sm rounded-full cursor-pointer !rounded-button ${darkMode ? 'bg-blue-600' : 'bg-blue-500'} text-white`}>3M</button>
                    <button className={`px-3 py-1 text-sm rounded-full cursor-pointer !rounded-button ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>6M</button>
                    <button className={`px-3 py-1 text-sm rounded-full cursor-pointer !rounded-button ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>1Y</button>
                    <button className={`px-3 py-1 text-sm rounded-full cursor-pointer !rounded-button ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>All</button>
                  </div>
                </div>
                <div id="portfolio-chart" className="w-full h-80"></div>
              </div>
              <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <h2 className="text-xl font-bold mb-4">Sentiment-Driven Recommendations</h2>
                <div className="space-y-4">
                  <div className={`p-4 rounded-lg ${darkMode ? 'bg-green-900 bg-opacity-20' : 'bg-green-50'} border-l-4 border-green-500`}>
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-bold">NVDA</h3>
                        <p className="text-sm text-gray-500">NVIDIA Corporation</p>
                      </div>
                      <div className="flex items-center">
                        <span className="text-green-500 font-bold mr-2">+0.89</span>
                        <i className="fas fa-arrow-up text-green-500"></i>
                      </div>
                    </div>
                    <p className="text-sm mt-2">Strong positive sentiment based on recent earnings call and AI advancements.</p>
                  </div>
                  <div className={`p-4 rounded-lg ${darkMode ? 'bg-green-900 bg-opacity-20' : 'bg-green-50'} border-l-4 border-green-500`}>
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-bold">AAPL</h3>
                        <p className="text-sm text-gray-500">Apple Inc.</p>
                      </div>
                      <div className="flex items-center">
                        <span className="text-green-500 font-bold mr-2">+0.78</span>
                        <i className="fas fa-arrow-up text-green-500"></i>
                      </div>
                    </div>
                    <p className="text-sm mt-2">Positive outlook due to upcoming product launches and strong consumer demand.</p>
                  </div>
                  <div className={`p-4 rounded-lg ${darkMode ? 'bg-red-900 bg-opacity-20' : 'bg-red-50'} border-l-4 border-red-500`}>
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="font-bold">TSLA</h3>
                        <p className="text-sm text-gray-500">Tesla, Inc.</p>
                      </div>
                      <div className="flex items-center">
                        <span className="text-red-500 font-bold mr-2">-0.35</span>
                        <i className="fas fa-arrow-down text-red-500"></i>
                      </div>
                    </div>
                    <p className="text-sm mt-2">Negative sentiment due to production challenges and increased competition.</p>
                  </div>
                  <button className="w-full py-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer whitespace-nowrap !rounded-button">
                    View all recommendations
                  </button>
                </div>
              </div>
            </div>
            {/* Sector Allocation and News Feed */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <h2 className="text-xl font-bold mb-4">Sector Allocation</h2>
                <div id="sector-allocation" className="w-full h-80"></div>
              </div>
              <div className={`lg:col-span-2 p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <h2 className="text-xl font-bold mb-4">Sentiment Analysis</h2>
                <div id="sentiment-chart" className="w-full h-80"></div>
              </div>
            </div>
            {/* News Feed */}
            <div className={`mt-6 p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <h2 className="text-xl font-bold mb-4">Latest Financial News</h2>
              <div className="space-y-4">
                <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-bold">Fed Signals Potential Rate Cut in Q3</h3>
                      <p className="text-sm text-gray-500">Financial Times • 2 hours ago</p>
                    </div>
                    <div className={`px-2 py-1 rounded text-xs ${darkMode ? 'bg-green-900 text-green-400' : 'bg-green-100 text-green-800'}`}>
                      Bullish
                    </div>
                  </div>
                  <p className="text-sm mt-2">Federal Reserve Chair indicates potential interest rate cuts in Q3 2025 as inflation pressures ease.</p>
                </div>
                <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-bold">Tech Sector Leads Market Rally</h3>
                      <p className="text-sm text-gray-500">Bloomberg • 5 hours ago</p>
                    </div>
                    <div className={`px-2 py-1 rounded text-xs ${darkMode ? 'bg-green-900 text-green-400' : 'bg-green-100 text-green-800'}`}>
                      Bullish
                    </div>
                  </div>
                  <p className="text-sm mt-2">Technology stocks continue to outperform broader market on strong earnings and AI advancements.</p>
                </div>
                <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-bold">Oil Prices Fall on Supply Concerns</h3>
                      <p className="text-sm text-gray-500">Reuters • Yesterday</p>
                    </div>
                    <div className={`px-2 py-1 rounded text-xs ${darkMode ? 'bg-red-900 text-red-400' : 'bg-red-100 text-red-800'}`}>
                      Bearish
                    </div>
                  </div>
                  <p className="text-sm mt-2">Crude oil prices drop as OPEC+ considers increasing production amid global economic slowdown fears.</p>
                </div>
                <button className="w-full py-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer whitespace-nowrap !rounded-button">
                  View all news
                </button>
              </div>
            </div>
          </div>
        )}
        {activeTab === 'sentiment' && (
          <div className="container mx-auto px-4 py-6">
            <div className={`p-6 rounded-lg shadow-md mb-6 ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <h2 className="text-xl font-bold mb-4">Sentiment Analysis Tool</h2>
              <div className="mb-4">
                <div className="relative">
                  <input
                    id="stock-input"
                    type="text"
                    placeholder="Enter stock symbol or company name..."
                    className={`w-full px-4 py-3 rounded-lg border-none text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}
                  />
                  <button 
                    id="analyze-btn"
                    onClick={() => {
                      const stockInput = document.getElementById('stock-input') as HTMLInputElement;
                      const stockSymbol = stockInput?.value.trim().toUpperCase();
                      
                      if (!stockSymbol) {
                        const validationMsg = document.createElement('div');
                        validationMsg.className = `fixed bottom-4 right-4 px-6 py-3 rounded-lg ${darkMode ? 'bg-red-900' : 'bg-red-100'} text-red-700 z-50`;
                        validationMsg.textContent = 'Please enter a valid stock symbol';
                        document.body.appendChild(validationMsg);
                        setTimeout(() => document.body.removeChild(validationMsg), 3000);
                        return;
                      }

                      // Show loading indicator
                      const loadingIndicator = document.createElement('div');
                      loadingIndicator.id = 'loading-indicator';
                      loadingIndicator.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
                      loadingIndicator.innerHTML = `
                        <div class="bg-white rounded-lg p-6 flex items-center space-x-4">
                          <div class="animate-spin rounded-full h-8 w-8 border-4 border-blue-600 border-t-transparent"></div>
                          <span>Analyzing ${stockSymbol}...</span>
                        </div>
                      `;
                      document.body.appendChild(loadingIndicator);

                      // Simulate API call delay
                      setTimeout(() => {
                        // Remove loading indicator
                        const indicator = document.getElementById('loading-indicator');
                        if (indicator) document.body.removeChild(indicator);

                        // Update sentiment results
                        const resultsContainer = document.getElementById('sentiment-results');
                        if (resultsContainer) {
                          resultsContainer.style.display = 'block';
                          
                          // Update sentiment score
                          const scoreElement = document.getElementById('sentiment-score');
                          if (scoreElement) scoreElement.textContent = '0.78';

                          // Update sentiment sources
                          const newsScoreElement = document.getElementById('news-sentiment');
                          const socialScoreElement = document.getElementById('social-sentiment');
                          const earningsScoreElement = document.getElementById('earnings-sentiment');
                          
                          if (newsScoreElement) newsScoreElement.textContent = '+0.82';
                          if (socialScoreElement) socialScoreElement.textContent = '+0.75';
                          if (earningsScoreElement) earningsScoreElement.textContent = '+0.68';
                        }

                        // Show success message
                        const successMsg = document.createElement('div');
                        successMsg.className = `fixed bottom-4 right-4 px-6 py-3 rounded-lg ${darkMode ? 'bg-green-900' : 'bg-green-100'} text-green-700 z-50`;
                        successMsg.textContent = `Analysis completed for ${stockSymbol}`;
                        document.body.appendChild(successMsg);
                        setTimeout(() => document.body.removeChild(successMsg), 3000);
                      }, 2000);
                    }}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 px-4 py-1 bg-blue-600 text-white rounded-full text-sm cursor-pointer !rounded-button"
                  >
                    Analyze
                  </button>
                </div>
              </div>
              <div className="flex flex-wrap -mx-2">
                <div className="w-full md:w-1/3 px-2 mb-4">
                  <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'} h-full`}>
                    <h3 className="text-lg font-semibold mb-3">Overall Sentiment</h3>
                    <div className="flex justify-center mb-4">
                      <div className="relative w-40 h-40">
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="text-center">
                            <div id="sentiment-score" className="text-3xl font-bold text-green-500">0.78</div>
                            <div className="text-sm text-gray-500">Strongly Bullish</div>
                          </div>
                        </div>
                        <svg className="w-full h-full" viewBox="0 0 100 100">
                          <circle
                            cx="50"
                            cy="50"
                            r="45"
                            fill="none"
                            stroke={darkMode ? "#374151" : "#E5E7EB"}
                            strokeWidth="10"
                          />
                          <circle
                            cx="50"
                            cy="50"
                            r="45"
                            fill="none"
                            stroke="#10B981"
                            strokeWidth="10"
                            strokeDasharray="282.7"
                            strokeDashoffset="62.2"
                            transform="rotate(-90 50 50)"
                          />
                        </svg>
                      </div>
                    </div>
                    <div id="sentiment-results" style={{display: 'none'}} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">News Sentiment</span>
                        <span id="news-sentiment" className="text-sm text-green-500">+0.82</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Social Media</span>
                        <span id="social-sentiment" className="text-sm text-green-500">+0.75</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Earnings Call</span>
                        <span id="earnings-sentiment" className="text-sm text-green-500">+0.68</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="w-full md:w-2/3 px-2 mb-4">
                <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'} h-full`}>
                  <h3 className="text-lg font-semibold mb-3">Sentiment Timeline</h3>
                  <div className="h-64 w-full">
                    <img
                      src="https://readdy.ai/api/search-image?query=A%20professional%20financial%20chart%20showing%20sentiment%20analysis%20timeline%20with%20a%20blue%20line%20trending%20upward%20over%20time%2C%20with%20clear%20data%20points%20and%20grid%20lines%2C%20clean%20modern%20design%20on%20a%20light%20background%2C%20suitable%20for%20financial%20dashboard&width=800&height=400&seq=1&orientation=landscape"
                      alt="Sentiment Timeline Chart"
                      className="w-full h-full object-cover object-top rounded-lg"
                    />
                  </div>
                </div>
              </div>
            </div>
            <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold">News Analysis</h2>
                <div className="flex space-x-2">
                  <button className={`px-3 py-1 text-sm rounded-full cursor-pointer !rounded-button ${darkMode ? 'bg-blue-600' : 'bg-blue-500'} text-white`}>All</button>
                  <button className={`px-3 py-1 text-sm rounded-full cursor-pointer !rounded-button ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>Bullish</button>
                  <button className={`px-3 py-1 text-sm rounded-full cursor-pointer !rounded-button ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>Bearish</button>
                  <button className={`px-3 py-1 text-sm rounded-full cursor-pointer !rounded-button ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>Neutral</button>
                </div>
              </div>
              <div className="space-y-4">
                <div className={`p-4 rounded-lg ${darkMode ? 'bg-green-900 bg-opacity-20' : 'bg-green-50'} border-l-4 border-green-500`}>
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-bold">Apple's New AI Features Boost iPhone Demand</h3>
                      <p className="text-sm text-gray-500">Wall Street Journal • May 18, 2025</p>
                    </div>
                    <div className="flex items-center">
                      <span className="text-green-500 font-bold mr-2">+0.92</span>
                      <span className={`px-2 py-1 rounded text-xs ${darkMode ? 'bg-green-900 text-green-400' : 'bg-green-100 text-green-800'}`}>
                        Strong Bullish
                      </span>
                    </div>
                  </div>
                  <p className="text-sm mt-2">Apple's latest AI features have significantly boosted iPhone demand, with analysts projecting a 15% increase in sales for the upcoming quarter.</p>
                  <div className="mt-3 flex justify-between items-center">
                    <div className="flex items-center">
                      <span className="text-xs text-gray-500 mr-2">Source Credibility:</span>
                      <div className="flex">
                        <i className="fas fa-star text-yellow-400 text-xs"></i>
                        <i className="fas fa-star text-yellow-400 text-xs"></i>
                        <i className="fas fa-star text-yellow-400 text-xs"></i>
                        <i className="fas fa-star text-yellow-400 text-xs"></i>
                        <i className="fas fa-star text-yellow-400 text-xs"></i>
                      </div>
                    </div>
                    <button className="text-blue-600 text-sm cursor-pointer whitespace-nowrap !rounded-button">Read full article</button>
                  </div>
                </div>
                <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'} border-l-4 border-gray-400`}>
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-bold">Apple's Q2 Earnings Meet Expectations</h3>
                      <p className="text-sm text-gray-500">Reuters • May 15, 2025</p>
                    </div>
                    <div className="flex items-center">
                      <span className="text-gray-500 font-bold mr-2">+0.05</span>
                      <span className={`px-2 py-1 rounded text-xs ${darkMode ? 'bg-gray-600 text-gray-300' : 'bg-gray-200 text-gray-700'}`}>
                        Neutral
                      </span>
                    </div>
                  </div>
                  <p className="text-sm mt-2">Apple reported Q2 earnings in line with analyst expectations, with revenue of $98.5 billion and EPS of $1.56.</p>
                  <div className="mt-3 flex justify-between items-center">
                    <div className="flex items-center">
                      <span className="text-xs text-gray-500 mr-2">Source Credibility:</span>
                      <div className="flex">
                        <i className="fas fa-star text-yellow-400 text-xs"></i>
                        <i className="fas fa-star text-yellow-400 text-xs"></i>
                        <i className="fas fa-star text-yellow-400 text-xs"></i>
                        <i className="fas fa-star text-yellow-400 text-xs"></i>
                        <i className="fas fa-star text-gray-300 text-xs"></i>
                      </div>
                    </div>
                    <button className="text-blue-600 text-sm cursor-pointer whitespace-nowrap !rounded-button">Read full article</button>
                  </div>
                </div>
                <div className={`p-4 rounded-lg ${darkMode ? 'bg-red-900 bg-opacity-20' : 'bg-red-50'} border-l-4 border-red-500`}>
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-bold">Supply Chain Issues May Impact iPhone Production</h3>
                      <p className="text-sm text-gray-500">Bloomberg • May 12, 2025</p>
                    </div>
                    <div className="flex items-center">
                      <span className="text-red-500 font-bold mr-2">-0.38</span>
                      <span className={`px-2 py-1 rounded text-xs ${darkMode ? 'bg-red-900 text-red-400' : 'bg-red-100 text-red-800'}`}>
                        Bearish
                      </span>
                    </div>
                  </div>
                  <p className="text-sm mt-2">Ongoing supply chain constraints in Asia could impact Apple's iPhone production capacity for the next quarter, potentially delaying new model launches.</p>
                  <div className="mt-3 flex justify-between items-center">
                    <div className="flex items-center">
                      <span className="text-xs text-gray-500 mr-2">Source Credibility:</span>
                      <div className="flex">
                        <i className="fas fa-star text-yellow-400 text-xs"></i>
                        <i className="fas fa-star text-yellow-400 text-xs"></i>
                        <i className="fas fa-star text-yellow-400 text-xs"></i>
                        <i className="fas fa-star text-yellow-400 text-xs"></i>
                        <i className="fas fa-star text-gray-300 text-xs"></i>
                      </div>
                    </div>
                    <button className="text-blue-600 text-sm cursor-pointer whitespace-nowrap !rounded-button">Read full article</button>
                  </div>
                </div>
                <button className="w-full py-2 text-sm text-blue-600 hover:text-blue-800 cursor-pointer whitespace-nowrap !rounded-button">
                  Load more news
                </button>
              </div>
            </div>
          </div>
        )}
        {activeTab === 'fundamental' && (
          <div className="container mx-auto px-4 py-6">
            <div className={`p-6 rounded-lg shadow-md mb-6 ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <h2 className="text-xl font-bold mb-4">Fundamental Analysis</h2>
              <div className="mb-4">
                <div className="relative">
                  <input
                    id="stock-input"
                    type="text"
                    placeholder="Enter stock symbol or company name..."
                    className={`w-full px-4 py-3 rounded-lg border-none text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}
                  />
                  <button 
                    id="analyze-btn"
                    onClick={() => {
                      const stockInput = document.getElementById('stock-input') as HTMLInputElement;
                      const stockSymbol = stockInput?.value.trim().toUpperCase();
                      
                      if (!stockSymbol) {
                        const validationMsg = document.createElement('div');
                        validationMsg.className = `fixed bottom-4 right-4 px-6 py-3 rounded-lg ${darkMode ? 'bg-red-900' : 'bg-red-100'} text-red-700 z-50`;
                        validationMsg.textContent = 'Please enter a valid stock symbol';
                        document.body.appendChild(validationMsg);
                        setTimeout(() => document.body.removeChild(validationMsg), 3000);
                        return;
                      }

                      // Show loading indicator
                      const loadingIndicator = document.createElement('div');
                      loadingIndicator.id = 'loading-indicator';
                      loadingIndicator.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
                      loadingIndicator.innerHTML = `
                        <div class="bg-white rounded-lg p-6 flex items-center space-x-4">
                          <div class="animate-spin rounded-full h-8 w-8 border-4 border-blue-600 border-t-transparent"></div>
                          <span>Analyzing ${stockSymbol}...</span>
                        </div>
                      `;
                      document.body.appendChild(loadingIndicator);

                      // Simulate API call delay
                      setTimeout(() => {
                        // Remove loading indicator
                        const indicator = document.getElementById('loading-indicator');
                        if (indicator) document.body.removeChild(indicator);

                        // Update sentiment results
                        const resultsContainer = document.getElementById('sentiment-results');
                        if (resultsContainer) {
                          resultsContainer.style.display = 'block';
                          
                          // Update sentiment score
                          const scoreElement = document.getElementById('sentiment-score');
                          if (scoreElement) scoreElement.textContent = '0.78';

                          // Update sentiment sources
                          const newsScoreElement = document.getElementById('news-sentiment');
                          const socialScoreElement = document.getElementById('social-sentiment');
                          const earningsScoreElement = document.getElementById('earnings-sentiment');
                          
                          if (newsScoreElement) newsScoreElement.textContent = '+0.82';
                          if (socialScoreElement) socialScoreElement.textContent = '+0.75';
                          if (earningsScoreElement) earningsScoreElement.textContent = '+0.68';
                        }

                        // Show success message
                        const successMsg = document.createElement('div');
                        successMsg.className = `fixed bottom-4 right-4 px-6 py-3 rounded-lg ${darkMode ? 'bg-green-900' : 'bg-green-100'} text-green-700 z-50`;
                        successMsg.textContent = `Analysis completed for ${stockSymbol}`;
                        document.body.appendChild(successMsg);
                        setTimeout(() => document.body.removeChild(successMsg), 3000);
                      }, 2000);
                    }}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 px-4 py-1 bg-blue-600 text-white rounded-full text-sm cursor-pointer !rounded-button"
                  >
                    Analyze
                  </button>
                </div>
              </div>
              <div className="flex flex-col md:flex-row md:space-x-4">
                <div className="w-full md:w-1/3 mb-4 md:mb-0">
                  <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                    <div className="flex items-center mb-4">
                      <div className="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center mr-3">
                        <i className="fab fa-apple text-blue-600 text-xl"></i>
                      </div>
                      <div>
                        <h3 className="font-bold text-lg">Apple Inc.</h3>
                        <p className="text-sm text-gray-500">NASDAQ: AAPL</p>
                      </div>
                    </div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-500">Current Price</span>
                      <span className="font-bold">$182.63</span>
                    </div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-500">Market Cap</span>
                      <span className="font-medium">$2.85T</span>
                    </div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-500">P/E Ratio</span>
                      <span className="font-medium">28.5</span>
                    </div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-500">EPS (TTM)</span>
                      <span className="font-medium">$6.41</span>
                    </div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-500">Dividend Yield</span>
                      <span className="font-medium">0.52%</span>
                    </div>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm text-gray-500">52-Week Range</span>
                      <span className="font-medium">$142.18 - $198.23</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-500">Beta</span>
                      <span className="font-medium">1.28</span>
                    </div>
                  </div>
                </div>
                <div className="w-full md:w-2/3">
                  <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'} h-full`}>
                    <h3 className="text-lg font-semibold mb-3">AI-Generated Financial Insights</h3>
                    <div className="space-y-4">
                      <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                        <h4 className="font-medium mb-1">Revenue Growth</h4>
                        <p className="text-sm">Apple has maintained consistent revenue growth over the past 5 years, with a CAGR of 11.2%. The Services segment continues to be the fastest-growing business unit, now accounting for 23.5% of total revenue.</p>
                      </div>
                      <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                        <h4 className="font-medium mb-1">Profitability</h4>
                        <p className="text-sm">Gross margin has expanded to 43.8%, above the 5-year average of 40.2%. Operating margin remains strong at 30.3%, reflecting efficient cost management despite inflationary pressures.</p>
                      </div>
                      <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                        <h4 className="font-medium mb-1">Balance Sheet Strength</h4>
                        <p className="text-sm">Apple maintains a robust balance sheet with $62.5B in cash and short-term investments. The debt-to-equity ratio of 1.68 is manageable given the company's strong cash flow generation.</p>
                      </div>
                      <div className={`p-3 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                        <h4 className="font-medium mb-1">Valuation Assessment</h4>
                        <p className="text-sm">At current levels, AAPL trades at a premium to the S&P 500 (28.5x vs. 22.1x forward P/E). However, this premium is justified by the company's strong brand, ecosystem advantages, and consistent capital return program.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <h2 className="text-xl font-bold mb-4">Financial Ratios</h2>
                <div className="space-y-4">
                  <div className="flex items-center">
                    <div className="w-1/2">
                      <div className="flex items-center">
                        <span className="text-sm mr-2">Profitability</span>
                        <i className="fas fa-info-circle text-gray-400 text-xs"></i>
                      </div>
                    </div>
                    <div className="w-1/2 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div className="h-full bg-green-500 rounded-full" style={{width: '85%'}}></div>
                    </div>
                  </div>
                  <div className="flex items-center">
                    <div className="w-1/2">
                      <div className="flex items-center">
                        <span className="text-sm mr-2">Growth</span>
                        <i className="fas fa-info-circle text-gray-400 text-xs"></i>
                      </div>
                    </div>
                    <div className="w-1/2 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div className="h-full bg-blue-500 rounded-full" style={{width: '72%'}}></div>
                    </div>
                  </div>
                  <div className="flex items-center">
                    <div className="w-1/2">
                      <div className="flex items-center">
                        <span className="text-sm mr-2">Financial Health</span>
                        <i className="fas fa-info-circle text-gray-400 text-xs"></i>
                      </div>
                    </div>
                    <div className="w-1/2 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div className="h-full bg-green-500 rounded-full" style={{width: '90%'}}></div>
                    </div>
                  </div>
                  <div className="flex items-center">
                    <div className="w-1/2">
                      <div className="flex items-center">
                        <span className="text-sm mr-2">Valuation</span>
                        <i className="fas fa-info-circle text-gray-400 text-xs"></i>
                      </div>
                    </div>
                    <div className="w-1/2 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div className="h-full bg-yellow-500 rounded-full" style={{width: '60%'}}></div>
                    </div>
                  </div>
                  <div className="flex items-center">
                    <div className="w-1/2">
                      <div className="flex items-center">
                        <span className="text-sm mr-2">Dividend</span>
                        <i className="fas fa-info-circle text-gray-400 text-xs"></i>
                      </div>
                    </div>
                    <div className="w-1/2 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div className="h-full bg-yellow-500 rounded-full" style={{width: '45%'}}></div>
                    </div>
                  </div>
                </div>
              </div>
              <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <h2 className="text-xl font-bold mb-4">Competitive Analysis</h2>
                <div className="overflow-x-auto">
                  <table className="min-w-full">
                    <thead>
                      <tr className={`${darkMode ? 'border-gray-700' : 'border-gray-200'} border-b`}>
                        <th className="text-left text-sm font-medium py-2">Metric</th>
                        <th className="text-center text-sm font-medium py-2">AAPL</th>
                        <th className="text-center text-sm font-medium py-2">MSFT</th>
                        <th className="text-center text-sm font-medium py-2">GOOGL</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className={`${darkMode ? 'border-gray-700' : 'border-gray-200'} border-b`}>
                        <td className="py-2 text-sm">P/E Ratio</td>
                        <td className="py-2 text-sm text-center">28.5</td>
                        <td className="py-2 text-sm text-center">34.2</td>
                        <td className="py-2 text-sm text-center">25.1</td>
                      </tr>
                      <tr className={`${darkMode ? 'border-gray-700' : 'border-gray-200'} border-b`}>
                        <td className="py-2 text-sm">Revenue Growth (YoY)</td>
                        <td className="py-2 text-sm text-center">8.1%</td>
                        <td className="py-2 text-sm text-center">12.4%</td>
                        <td className="py-2 text-sm text-center">15.2%</td>
                      </tr>
                      <tr className={`${darkMode ? 'border-gray-700' : 'border-gray-200'} border-b`}>
                        <td className="py-2 text-sm">Gross Margin</td>
                        <td className="py-2 text-sm text-center">43.8%</td>
                        <td className="py-2 text-sm text-center">68.2%</td>
                        <td className="py-2 text-sm text-center">56.9%</td>
                      </tr>
                      <tr className={`${darkMode ? 'border-gray-700' : 'border-gray-200'} border-b`}>
                        <td className="py-2 text-sm">ROE</td>
                        <td className="py-2 text-sm text-center">147.9%</td>
                        <td className="py-2 text-sm text-center">42.5%</td>
                        <td className="py-2 text-sm text-center">28.6%</td>
                      </tr>
                      <tr>
                        <td className="py-2 text-sm">Dividend Yield</td>
                        <td className="py-2 text-sm text-center">0.52%</td>
                        <td className="py-2 text-sm text-center">0.75%</td>
                        <td className="py-2 text-sm text-center">-</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <h2 className="text-xl font-bold mb-4">Financial Statement Analysis</h2>
              <div className="mb-4">
                <div className="flex space-x-2 mb-4">
                  <button className={`px-3 py-1 text-sm rounded-full cursor-pointer !rounded-button ${darkMode ? 'bg-blue-600' : 'bg-blue-500'} text-white`}>Income Statement</button>
                  <button className={`px-3 py-1 text-sm rounded-full cursor-pointer !rounded-button ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>Balance Sheet</button>
                  <button className={`px-3 py-1 text-sm rounded-full cursor-pointer !rounded-button ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>Cash Flow</button>
                </div>
                <div className="overflow-x-auto">
                  <table className="min-w-full">
                    <thead>
                      <tr className={`${darkMode ? 'border-gray-700' : 'border-gray-200'} border-b`}>
                        <th className="text-left text-sm font-medium py-2">Metric ($ Millions)</th>
                        <th className="text-right text-sm font-medium py-2">2025 Q1</th>
                        <th className="text-right text-sm font-medium py-2">2024 Q4</th>
                        <th className="text-right text-sm font-medium py-2">2024 Q3</th>
                        <th className="text-right text-sm font-medium py-2">2024 Q2</th>
                        <th className="text-right text-sm font-medium py-2">YoY Change</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className={`${darkMode ? 'border-gray-700' : 'border-gray-200'} border-b`}>
                        <td className="py-2 text-sm">Revenue</td>
                        <td className="py-2 text-sm text-right">$98,522</td>
                        <td className="py-2 text-sm text-right">$119,575</td>
                        <td className="py-2 text-sm text-right">$89,498</td>
                        <td className="py-2 text-sm text-right">$94,836</td>
                        <td className="py-2 text-sm text-right text-green-500">+8.1%</td>
                      </tr>
                      <tr className={`${darkMode ? 'border-gray-700' : 'border-gray-200'} border-b`}>
                        <td className="py-2 text-sm">Gross Profit</td>
                        <td className="py-2 text-sm text-right">$43,156</td>
                        <td className="py-2 text-sm text-right">$52,606</td>
                        <td className="py-2 text-sm text-right">$38,099</td>
                        <td className="py-2 text-sm text-right">$40,425</td>
                        <td className="py-2 text-sm text-right text-green-500">+9.3%</td>
                      </tr>
                      <tr className={`${darkMode ? 'border-gray-700' : 'border-gray-200'} border-b`}>
                        <td className="py-2 text-sm">Operating Income</td>
                        <td className="py-2 text-sm text-right">$29,855</td>
                        <td className="py-2 text-sm text-right">$36,873</td>
                        <td className="py-2 text-sm text-right">$25,359</td>
                        <td className="py-2 text-sm text-right">$28,653</td>
                        <td className="py-2 text-sm text-right text-green-500">+7.5%</td>
                      </tr>
                      <tr className={`${darkMode ? 'border-gray-700' : 'border-gray-200'} border-b`}>
                        <td className="py-2 text-sm">Net Income</td>
                        <td className="py-2 text-sm text-right">$24,159</td>
                        <td className="py-2 text-sm text-right">$29,998</td>
                        <td className="py-2 text-sm text-right">$20,789</td>
                        <td className="py-2 text-sm text-right">$23,629</td>
                        <td className="py-2 text-sm text-right text-green-500">+6.8%</td>
                      </tr>
                      <tr className={`${darkMode ? 'border-gray-700' : 'border-gray-200'} border-b`}>
                        <td className="py-2 text-sm">EPS (Diluted)</td>
                        <td className="py-2 text-sm text-right">$1.56</td>
                        <td className="py-2 text-sm text-right">$1.92</td>
                        <td className="py-2 text-sm text-right">$1.33</td>
                        <td className="py-2 text-sm text-right">$1.52</td>
                        <td className="py-2 text-sm text-right text-green-500">+7.6%</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
              <div className="mt-6">
                <h3 className="text-lg font-semibold mb-3">AI Financial Analysis</h3>
                <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                  <p className="text-sm mb-3">Apple's Q1 2025 results demonstrate continued resilience despite macroeconomic headwinds. Key observations:</p>
                  <ul className="list-disc pl-5 text-sm space-y-2">
                    <li>Revenue growth of 8.1% YoY outpaced the overall tech sector average of 5.3%</li>
                    <li>Gross margin expansion of 50 basis points indicates pricing power and operational efficiency</li>
                    <li>Services revenue grew 18.2% YoY, now representing 23.5% of total revenue</li>
                    <li>iPhone revenue increased 5.3% YoY, showing continued demand strength despite elongating replacement cycles</li>
                    <li>Operating expenses as a percentage of revenue decreased by 30 basis points, reflecting disciplined cost management</li>
                  </ul>
                  <p className="text-sm mt-3">Overall, Apple continues to execute well on its strategic priorities while navigating supply chain challenges and evolving consumer preferences. The company's strong cash position and consistent capital return program provide downside protection.</p>
                </div>
              </div>
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
                    <p className="text-sm text-gray-500 mb-4">Upload a screenshot or CSV file of your current portfolio</p>
                    <input
                      type="file"
                      id="portfolio-file-input"
                      accept=".png,.jpg,.pdf,.csv"
                      className="hidden"
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) {
                          const reader = new FileReader();
                          reader.onload = (event) => {
                            const fileContent = event.target?.result || null;
                            setUploadedFile({
                              name: file.name,
                              size: (file.size / 1024).toFixed(2),
                              type: file.type,
                              content: fileContent
                            });
                            setShowPreview(true);
                          };
                          if (file.type.includes('image')) {
                            reader.readAsDataURL(file);
                          } else {
                            reader.readAsText(file);
                          }
                        }
                      }}
                    />
                    <button
                      onClick={() => document.getElementById('portfolio-file-input')?.click()}
                      className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm cursor-pointer !rounded-button"
                    >
                      Select File
                    </button>
                    <p className="text-xs text-gray-500 mt-2">Supported formats: PNG, JPG, PDF, CSV</p>
                  </div>
                </div>
                {/* File Preview Modal */}
                {showPreview && uploadedFile && (
                  <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className={`max-w-2xl w-full mx-4 p-6 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                      <h3 className="text-xl font-bold mb-4">File Preview</h3>
                      <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'} mb-4`}>
                        <div className="flex items-center mb-3">
                          <i className="fas fa-file text-blue-500 text-2xl mr-3"></i>
                          <div>
                            <p className="font-medium">{uploadedFile.name}</p>
                            <p className="text-sm text-gray-500">{uploadedFile.size} KB</p>
                          </div>
                        </div>
                        {uploadedFile.type.includes('image') ? (
                          <img
                            src={uploadedFile.content as string}
                            alt="Portfolio Preview"
                            className="max-h-96 w-full object-contain rounded"
                          />
                        ) : (
                          <div className={`p-4 rounded ${darkMode ? 'bg-gray-800' : 'bg-gray-200'} max-h-96 overflow-auto`}>
                            <pre className="text-sm whitespace-pre-wrap">{uploadedFile.content as string}</pre>
                          </div>
                        )}
                      </div>
                      <div className="flex justify-end space-x-3">
                        <button
                          onClick={() => {
                            setShowPreview(false);
                            setUploadedFile(null);
                          }}
                          className={`px-4 py-2 rounded-md text-sm cursor-pointer !rounded-button ${darkMode ? 'bg-gray-700 text-white' : 'bg-gray-200 text-gray-800'}`}
                        >
                          Cancel
                        </button>
                        <button
                          onClick={() => {
                            setShowPreview(false);
                            // Process the file and update portfolio data
                            processPortfolioFile(uploadedFile);
                          }}
                          className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm cursor-pointer !rounded-button"
                        >
                          Confirm Upload
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold mb-3">Current Portfolio</h3>
                  <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                    <div className="mb-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm">Total Value</span>
                        <span className="font-bold">$35,500.00</span>
                      </div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm">Annualized Return</span>
                        <span className="font-medium text-green-500">+12.4%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Risk Level</span>
                        <span className="font-medium text-yellow-500">Moderate</span>
                      </div>
                    </div>
                    <div className="mb-4">
                      <h4 className="text-sm font-medium mb-2">Asset Allocation</h4>
                      <div className="h-4 w-full rounded-full overflow-hidden flex">
                        <div className="h-full bg-blue-500" style={{width: '35%'}}></div>
                        <div className="h-full bg-green-500" style={{width: '20%'}}></div>
                        <div className="h-full bg-yellow-500" style={{width: '15%'}}></div>
                        <div className="h-full bg-purple-500" style={{width: '12%'}}></div>
                        <div className="h-full bg-red-500" style={{width: '10%'}}></div>
                        <div className="h-full bg-gray-500" style={{width: '8%'}}></div>
                      </div>
                      <div className="flex flex-wrap mt-2">
                        <div className="flex items-center mr-4 mb-1">
                          <div className="w-3 h-3 bg-blue-500 rounded-full mr-1"></div>
                          <span className="text-xs">Technology (35%)</span>
                        </div>
                        <div className="flex items-center mr-4 mb-1">
                          <div className="w-3 h-3 bg-green-500 rounded-full mr-1"></div>
                          <span className="text-xs">Healthcare (20%)</span>
                        </div>
                        <div className="flex items-center mr-4 mb-1">
                          <div className="w-3 h-3 bg-yellow-500 rounded-full mr-1"></div>
                          <span className="text-xs">Consumer (15%)</span>
                        </div>
                        <div className="flex items-center mr-4 mb-1">
                          <div className="w-3 h-3 bg-purple-500 rounded-full mr-1"></div>
                          <span className="text-xs">Financial (12%)</span>
                        </div>
                        <div className="flex items-center mr-4 mb-1">
                          <div className="w-3 h-3 bg-red-500 rounded-full mr-1"></div>
                          <span className="text-xs">Communication (10%)</span>
                        </div>
                        <div className="flex items-center mb-1">
                          <div className="w-3 h-3 bg-gray-500 rounded-full mr-1"></div>
                          <span className="text-xs">Others (8%)</span>
                        </div>
                      </div>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium mb-2">Top Holdings</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <div className="flex items-center">
                            <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center mr-2">
                              <i className="fab fa-apple text-blue-600"></i>
                            </div>
                            <div>
                              <p className="text-sm font-medium">Apple Inc.</p>
                              <p className="text-xs text-gray-500">AAPL</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="text-sm font-medium">15.2%</p>
                            <p className="text-xs text-green-500">+8.3%</p>
                          </div>
                        </div>
                        <div className="flex justify-between items-center">
                          <div className="flex items-center">
                            <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center mr-2">
                              <i className="fab fa-microsoft text-blue-600"></i>
                            </div>
                            <div>
                              <p className="text-sm font-medium">Microsoft Corp.</p>
                              <p className="text-xs text-gray-500">MSFT</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="text-sm font-medium">12.8%</p>
                            <p className="text-xs text-green-500">+15.2%</p>
                          </div>
                        </div>
                        <div className="flex justify-between items-center">
                          <div className="flex items-center">
                            <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center mr-2">
                              <i className="fas fa-pills text-green-600"></i>
                            </div>
                            <div>
                              <p className="text-sm font-medium">Johnson & Johnson</p>
                              <p className="text-xs text-gray-500">JNJ</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="text-sm font-medium">8.5%</p>
                            <p className="text-xs text-red-500">-2.1%</p>
                          </div>
                        </div>
                        <div className="flex justify-between items-center">
                          <div className="flex items-center">
                            <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center mr-2">
                              <i className="fab fa-amazon text-blue-600"></i>
                            </div>
                            <div>
                              <p className="text-sm font-medium">Amazon.com Inc.</p>
                              <p className="text-xs text-gray-500">AMZN</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="text-sm font-medium">7.3%</p>
                            <p className="text-xs text-green-500">+5.8%</p>
                          </div>
                        </div>
                        <div className="flex justify-between items-center">
                          <div className="flex items-center">
                            <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center mr-2">
                              <i className="fab fa-google text-blue-600"></i>
                            </div>
                            <div>
                              <p className="text-sm font-medium">Alphabet Inc.</p>
                              <p className="text-xs text-gray-500">GOOGL</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="text-sm font-medium">6.9%</p>
                            <p className="text-xs text-green-500">+10.2%</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                <div>
                  <h3 className="text-lg font-semibold mb-3">AI-Recommended Optimization</h3>
                  <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                    <div className="mb-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm">Projected Value (1Y)</span>
                        <span className="font-bold">$41,230.00</span>
                      </div>
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm">Projected Return</span>
                        <span className="font-medium text-green-500">+16.1%</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Optimized Risk Level</span>
                        <span className="font-medium text-blue-500">Moderate-Low</span>
                      </div>
                    </div>
                    <div className="mb-4">
                      <h4 className="text-sm font-medium mb-2">Recommended Asset Allocation</h4>
                      <div className="h-4 w-full rounded-full overflow-hidden flex">
                        <div className="h-full bg-blue-500" style={{width: '30%'}}></div>
                        <div className="h-full bg-green-500" style={{width: '25%'}}></div>
                        <div className="h-full bg-yellow-500" style={{width: '15%'}}></div>
                        <div className="h-full bg-purple-500" style={{width: '15%'}}></div>
                        <div className="h-full bg-red-500" style={{width: '8%'}}></div>
                        <div className="h-full bg-gray-500" style={{width: '7%'}}></div>
                      </div>
                      <div className="flex flex-wrap mt-2">
                        <div className="flex items-center mr-4 mb-1">
                          <div className="w-3 h-3 bg-blue-500 rounded-full mr-1"></div>
                          <span className="text-xs">Technology (30%)</span>
                        </div>
                        <div className="flex items-center mr-4 mb-1">
                          <div className="w-3 h-3 bg-green-500 rounded-full mr-1"></div>
                          <span className="text-xs">Healthcare (25%)</span>
                        </div>
                        <div className="flex items-center mr-4 mb-1">
                          <div className="w-3 h-3 bg-yellow-500 rounded-full mr-1"></div>
                          <span className="text-xs">Consumer (15%)</span>
                        </div>
                        <div className="flex items-center mr-4 mb-1">
                          <div className="w-3 h-3 bg-purple-500 rounded-full mr-1"></div>
                          <span className="text-xs">Financial (15%)</span>
                        </div>
                        <div className="flex items-center mr-4 mb-1">
                          <div className="w-3 h-3 bg-red-500 rounded-full mr-1"></div>
                          <span className="text-xs">Communication (8%)</span>
                        </div>
                        <div className="flex items-center mb-1">
                          <div className="w-3 h-3 bg-gray-500 rounded-full mr-1"></div>
                          <span className="text-xs">Others (7%)</span>
                        </div>
                      </div>
                    </div>
                    <div className="mb-4">
                      <h4 className="text-sm font-medium mb-2">Recommended Actions</h4>
                      <div className="space-y-3">
                        <div className={`p-3 rounded-lg ${darkMode ? 'bg-green-900 bg-opacity-20' : 'bg-green-50'} border-l-4 border-green-500`}>
                          <p className="text-sm font-medium">Increase Healthcare Exposure</p>
                          <p className="text-xs mt-1">Increase allocation to healthcare sector from 20% to 25% to capitalize on positive sentiment and upcoming catalysts.</p>
                        </div>
                        <div className={`p-3 rounded-lg ${darkMode ? 'bg-red-900 bg-opacity-20' : 'bg-red-50'} border-l-4 border-red-500`}>
                          <p className="text-sm font-medium">Reduce Technology Exposure</p>
                          <p className="text-xs mt-1">Reduce technology sector allocation from 35% to 30% to mitigate concentration risk and take profits.</p>
                        </div>
                        <div className={`p-3 rounded-lg ${darkMode ? 'bg-green-900 bg-opacity-20' : 'bg-green-50'} border-l-4 border-green-500`}>
                          <p className="text-sm font-medium">Increase Financial Services</p>
                          <p className="text-xs mt-1">Increase financial services allocation from 12% to 15% to benefit from potential interest rate cuts.</p>
                        </div>
                      </div>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium mb-2">Tax-Loss Harvesting Opportunities</h4>
                      <div className={`p-3 rounded-lg ${darkMode ? 'bg-blue-900 bg-opacity-20' : 'bg-blue-50'} border-l-4 border-blue-500`}>
                        <p className="text-sm">Potential tax savings of approximately $420 identified through strategic tax-loss harvesting.</p>
                        <div className="mt-2 space-y-1">
                          <div className="flex justify-between text-xs">
                            <span>TSLA (Tesla Inc.)</span>
                            <span className="text-red-500">-$850 unrealized loss</span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span>NFLX (Netflix Inc.)</span>
                            <span className="text-red-500">-$320 unrealized loss</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div className="mt-4 flex justify-end">
                    <button
                      id="download-report-btn"
                      onClick={() => {
                        const dialog = document.getElementById('download-dialog');
                        if (dialog) dialog.style.display = 'flex';
                      }}
                      className="px-4 py-2 bg-gray-300 text-gray-800 rounded-md text-sm mr-2 cursor-pointer !rounded-button"
                    >
                      Download Report
                    </button>
                    {/* Download Dialog */}
                    <div
                      id="download-dialog"
                      className="fixed inset-0 bg-black bg-opacity-50 hidden items-center justify-center z-50"
                    >
                      <div className={`max-w-md w-full mx-4 p-6 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                        <h3 className="text-xl font-bold mb-4">Download Report</h3>
                        <div className="mb-4">
                          <label className="block text-sm font-medium mb-2">Format</label>
                          <div className="grid grid-cols-3 gap-2">
                            <button
                              id="pdf-format"
                              onClick={(e) => {
                                document.querySelectorAll('[id$="-format"]').forEach(btn =>
                                  btn.classList.remove('bg-blue-600', 'text-white'));
                                e.currentTarget.classList.add('bg-blue-600', 'text-white');
                              }}
                              className={`px-3 py-2 rounded-md text-sm cursor-pointer !rounded-button ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}
                            >
                              PDF
                            </button>
                            <button
                              id="excel-format"
                              onClick={(e) => {
                                document.querySelectorAll('[id$="-format"]').forEach(btn =>
                                  btn.classList.remove('bg-blue-600', 'text-white'));
                                e.currentTarget.classList.add('bg-blue-600', 'text-white');
                              }}
                              className={`px-3 py-2 rounded-md text-sm cursor-pointer !rounded-button ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}
                            >
                              Excel
                            </button>
                            <button
                              id="csv-format"
                              onClick={(e) => {
                                document.querySelectorAll('[id$="-format"]').forEach(btn =>
                                  btn.classList.remove('bg-blue-600', 'text-white'));
                                e.currentTarget.classList.add('bg-blue-600', 'text-white');
                              }}
                              className={`px-3 py-2 rounded-md text-sm cursor-pointer !rounded-button ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}
                            >
                              CSV
                            </button>
                          </div>
                        </div>
                        <div className="mb-6">
                          <label className="block text-sm font-medium mb-2">Include in Report</label>
                          <div className="space-y-2">
                            <label className="flex items-center">
                              <input type="checkbox" id="include-charts" className="mr-2" defaultChecked />
                              <span className="text-sm">Charts and Graphs</span>
                            </label>
                            <label className="flex items-center">
                              <input type="checkbox" id="include-recommendations" className="mr-2" defaultChecked />
                              <span className="text-sm">AI Recommendations</span>
                            </label>
                            <label className="flex items-center">
                              <input type="checkbox" id="include-tax" className="mr-2" defaultChecked />
                              <span className="text-sm">Tax Harvesting Opportunities</span>
                            </label>
                          </div>
                        </div>
                        <div className="flex justify-end space-x-3">
                          <button
                            onClick={() => {
                              const dialog = document.getElementById('download-dialog');
                              if (dialog) dialog.style.display = 'none';
                            }}
                            className={`px-4 py-2 rounded-md text-sm cursor-pointer !rounded-button ${darkMode ? 'bg-gray-700 text-white' : 'bg-gray-200 text-gray-800'}`}
                          >
                            Cancel
                          </button>
                          <button
                            onClick={() => {
                              const dialog = document.getElementById('download-dialog');
                              if (dialog) dialog.style.display = 'none';
                              // Show success message
                              const successMessage = document.createElement('div');
                              successMessage.className = `fixed bottom-4 right-4 px-6 py-3 rounded-lg ${darkMode ? 'bg-green-900' : 'bg-green-100'} text-green-700 z-50`;
                              successMessage.textContent = 'Report downloaded successfully!';
                              document.body.appendChild(successMessage);
                              // Remove success message after 3 seconds
                              setTimeout(() => {
                                document.body.removeChild(successMessage);
                              }, 3000);
                            }}
                            className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm cursor-pointer !rounded-button"
                          >
                            Download
                          </button>
                        </div>
                      </div>
                    </div>
                    <button className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm cursor-pointer !rounded-button">
                      Apply Recommendations
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
        <div className={`p-6 rounded-lg shadow-md ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
          <h2 className="text-xl font-bold mb-4">Risk Tolerance Adjustment</h2>
          <div className="mb-6">
            <h3 className="text-lg font-semibold mb-3">Current Risk Profile: <span className="text-yellow-500">Moderate</span></h3>
            <p className="text-sm mb-4">Adjust your risk tolerance to see how it affects your portfolio recommendations.</p>
            <div className="relative pt-1">
              <div className="flex justify-between mb-2">
                <span className="text-xs font-semibold text-blue-600">Conservative</span>
                <span className="text-xs font-semibold text-yellow-600">Moderate</span>
                <span className="text-xs font-semibold text-red-600">Aggressive</span>
              </div>
              <div className="h-2 w-full bg-gray-200 rounded-full">
                <div className="h-full bg-yellow-500 rounded-full" style={{width: '60%'}}></div>
              </div>
              <div className="absolute left-[60%] top-0 transform -translate-x-1/2">
                <div className="w-4 h-4 bg-white rounded-full border-2 border-yellow-500 cursor-pointer"></div>
              </div>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className={`p-4 rounded-lg ${darkMode ? 'bg-blue-900 bg-opacity-20' : 'bg-blue-50'} border-l-4 border-blue-500`}>
              <h3 className="font-bold mb-2">Conservative</h3>
              <ul className="text-sm space-y-2">
                <li className="flex items-start">
                  <i className="fas fa-check-circle text-blue-500 mt-0.5 mr-2"></i>
                  <span>Lower volatility</span>
                </li>
                <li className="flex items-start">
                  <i className="fas fa-check-circle text-blue-500 mt-0.5 mr-2"></i>
                  <span>Capital preservation focus</span>
                </li>
                <li className="flex items-start">
                  <i className="fas fa-check-circle text-blue-500 mt-0.5 mr-2"></i>
                  <span>Higher allocation to bonds and dividend stocks</span>
                </li>
                <li className="flex items-start">
                  <i className="fas fa-check-circle text-blue-500 mt-0.5 mr-2"></i>
                  <span>Projected return: 5-8%</span>
                </li>
              </ul>
            </div>
            <div className={`p-4 rounded-lg ${darkMode ? 'bg-yellow-900 bg-opacity-20' : 'bg-yellow-50'} border-l-4 border-yellow-500`}>
              <h3 className="font-bold mb-2">Moderate</h3>
              <ul className="text-sm space-y-2">
                <li className="flex items-start">
                  <i className="fas fa-check-circle text-yellow-500 mt-0.5 mr-2"></i>
                  <span>Balanced approach</span>
                </li>
                <li className="flex items-start">
                  <i className="fas fa-check-circle text-yellow-500 mt-0.5 mr-2"></i>
                  <span>Growth with reasonable risk</span>
                </li>
                <li className="flex items-start">
                  <i className="fas fa-check-circle text-yellow-500 mt-0.5 mr-2"></i>
                  <span>Diversified across stocks, bonds, and alternatives</span>
                </li>
                <li className="flex items-start">
                  <i className="fas fa-check-circle text-yellow-500 mt-0.5 mr-2"></i>
                  <span>Projected return: 8-12%</span>
                </li>
              </ul>
            </div>
            <div className={`p-4 rounded-lg ${darkMode ? 'bg-red-900 bg-opacity-20' : 'bg-red-50'} border-l-4 border-red-500`}>
              <h3 className="font-bold mb-2">Aggressive</h3>
              <ul className="text-sm space-y-2">
                <li className="flex items-start">
                  <i className="fas fa-check-circle text-red-500 mt-0.5 mr-2"></i>
                  <span>Higher volatility</span>
                </li>
                <li className="flex items-start">
                  <i className="fas fa-check-circle text-red-500 mt-0.5 mr-2"></i>
                  <span>Long-term growth focus</span>
                </li>
                <li className="flex items-start">
                  <i className="fas fa-check-circle text-red-500 mt-0.5 mr-2"></i>
                  <span>Higher allocation to growth stocks and emerging markets</span>
                </li>
                <li className="flex items-start">
                  <i className="fas fa-check-circle text-red-500 mt-0.5 mr-2"></i>
                  <span>Projected return: 12-18%</span>
                </li>
              </ul>
            </div>
          </div>
          <div className="mt-6">
            <button
              onClick={() => {
                const dialog = document.getElementById('risk-profile-dialog');
                if (dialog) dialog.style.display = 'flex';
              }}
              className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm cursor-pointer !rounded-button"
            >
              Update Risk Profile
            </button>
            {/* Risk Profile Update Dialog */}
            <div
              id="risk-profile-dialog"
              className="fixed inset-0 bg-black bg-opacity-50 hidden items-center justify-center z-50"
            >
              <div className={`max-w-md w-full mx-4 p-6 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <h3 className="text-xl font-bold mb-4">Update Risk Profile</h3>
                <div className="mb-6">
                  <p className="text-sm mb-4">Your risk profile will be updated from Moderate to Aggressive.</p>
                  <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-100'}`}>
                    <h4 className="font-medium mb-2">Potential Impact</h4>
                    <ul className="text-sm space-y-2">
                      <li className="flex items-start">
                        <i className="fas fa-arrow-right text-blue-500 mt-0.5 mr-2"></i>
                        <span>Portfolio allocation will shift towards higher growth assets</span>
                      </li>
                      <li className="flex items-start">
                        <i className="fas fa-arrow-right text-blue-500 mt-0.5 mr-2"></i>
                        <span>Expected returns may increase to 12-18% range</span>
                      </li>
                      <li className="flex items-start">
                        <i className="fas fa-arrow-right text-blue-500 mt-0.5 mr-2"></i>
                        <span>Portfolio volatility will likely increase</span>
                      </li>
                    </ul>
                  </div>
                </div>
                <div className="flex justify-end space-x-3">
                  <button
                    onClick={() => {
                      const dialog = document.getElementById('risk-profile-dialog');
                      if (dialog) dialog.style.display = 'none';
                    }}
                    className={`px-4 py-2 rounded-md text-sm cursor-pointer !rounded-button ${darkMode ? 'bg-gray-700 text-white' : 'bg-gray-200 text-gray-800'}`}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={() => {
                      const dialog = document.getElementById('risk-profile-dialog');
                      if (dialog) dialog.style.display = 'none';
                      // Show success message
                      const successMessage = document.createElement('div');
                      successMessage.className = `fixed bottom-4 right-4 px-6 py-3 rounded-lg ${darkMode ? 'bg-green-900' : 'bg-green-100'} text-green-700 z-50`;
                      successMessage.textContent = 'Risk profile updated successfully!';
                      document.body.appendChild(successMessage);
                      // Remove success message after 3 seconds
                      setTimeout(() => {
                        document.body.removeChild(successMessage);
                      }, 3000);
                    }}
                    className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm cursor-pointer !rounded-button"
                  >
                    Confirm Update
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;






// import React, { useState, useEffect, useCallback } from 'react';
// import * as echarts from 'echarts';
// import axios from 'axios';

// const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:3001/api';
// axios.get(`${API}/news`) 

// // ---------------- TYPES ---------------- //

// interface NewsArticle {
//   title: string;
//   source: string;
//   timestamp: string;
//   content: string;
//   url: string;
//   sentiment: number;
//   label: string;
//   confidence: number;
//   reasons: string[];
//   credibility: number;
// }

// interface StockData {
//   symbol: string;
//   name: string;
//   currentPrice: number;
//   sentiment?: string;
//   confidence?: number;
//   reasons?: string[];
// }

// // ---------------- APP ---------------- //

// const App: React.FC = () => {

//   const [activeTab, setActiveTab] = useState<string>('dashboard');
//   const [searchQuery, setSearchQuery] = useState<string>('');
//   const [newsData, setNewsData] = useState<NewsArticle[]>([]);
//   const [stockData, setStockData] = useState<StockData | null>(null);
//   const [loading, setLoading] = useState<boolean>(false);

//   // ---------------- FETCH ---------------- //

//   const fetchNewsData = useCallback(async () => {
//     try {
//       const res = await axios.get(`${API_BASE_URL}/news`);
//       setNewsData(res.data);
//     } catch (err) {
//       console.error(err);
//     }
//   }, []);

//   const fetchStockData = useCallback(async (symbol: string) => {
//     try {
//       setLoading(true);
//       const res = await axios.get(`${API_BASE_URL}/stock/${symbol}`);
//       setStockData(res.data);
//     } catch (err) {
//       console.error(err);
//     } finally {
//       setLoading(false);
//     }
//   }, []);

//   useEffect(() => {
//     fetchNewsData();
//   }, [fetchNewsData]);

//   // ---------------- CHART ---------------- //

//   useEffect(() => {
//     const chartEl = document.getElementById('sentiment-chart');
//     if (!chartEl || newsData.length === 0) return;

//     const chart = echarts.init(chartEl);

//     chart.setOption({
//       tooltip: {
//         trigger: 'axis',
//         formatter: function (params: any) {
//           const item = params[0];
//           const news = newsData[item.dataIndex];

//           return `
//             <b>${news.title}</b><br/>
//             Sentiment: ${news.label}<br/>
//             Confidence: ${news.confidence}%<br/>
//             Reasons: ${news.reasons.join(", ")}
//           `;
//         }
//       },
//       xAxis: {
//         type: 'category',
//         data: newsData.map((n) => n.title.slice(0, 10) + '...')
//       },
//       yAxis: {
//         type: 'value',
//         min: -1,
//         max: 1
//       },
//       series: [
//         {
//           type: 'bar',
//           data: newsData.map((n) => n.sentiment),
//           itemStyle: {
//             color: (params: any) =>
//               params.value > 0 ? '#10B981' : '#EF4444'
//           }
//         }
//       ]
//     });

//     return () => chart.dispose();
//   }, [newsData]);

//   // ---------------- UI ---------------- //

//   return (
//     <div className="min-h-screen bg-gray-50 text-gray-900 p-6">

//       {/* HEADER */}
//       <h1 className="text-3xl font-bold mb-6">SentiInvest</h1>

//       {/* NAV */}
//       <div className="flex gap-4 mb-6">
//         <button onClick={() => setActiveTab('dashboard')}>Dashboard</button>
//         <button onClick={() => setActiveTab('sentiment')}>Sentiment</button>
//       </div>

//       {/* SEARCH */}
//       <div className="mb-6">
//         <input
//           type="text"
//           placeholder="Enter stock ticker..."
//           className="border p-2 mr-2"
//           value={searchQuery}
//           onChange={(e) => setSearchQuery(e.target.value)}
//         />
//         <button
//           onClick={() => fetchStockData(searchQuery)}
//           className="bg-blue-500 text-white px-4 py-2"
//         >
//           Analyze
//         </button>
//       </div>

//       {/* STOCK RESULT */}
//       {stockData && (
//         <div className="p-4 bg-white rounded shadow mb-6">
//           <h2 className="text-xl font-bold">{stockData.name}</h2>
//           <p>Price: ${stockData.currentPrice}</p>

//           {stockData.sentiment && (
//             <>
//               <p>
//                 Sentiment:
//                 <span className={stockData.sentiment === 'Bullish' ? 'text-green-500' : 'text-red-500'}>
//                   {" "}{stockData.sentiment}
//                 </span>
//               </p>

//               <p>Confidence: {stockData.confidence}%</p>

//               <div>
//                 <p className="font-semibold">Why?</p>
//                 <ul className="list-disc ml-5">
//                   {stockData.reasons?.map((r, i) => (
//                     <li key={i}>{r}</li>
//                   ))}
//                 </ul>
//               </div>
//             </>
//           )}
//         </div>
//       )}

//       {/* SENTIMENT TAB */}
//       {activeTab === 'sentiment' && (
//         <>
//           <div id="sentiment-chart" style={{ height: 300 }} className="mb-8"></div>

//           <div className="grid gap-4">
//             {newsData.map((article, i) => (
//               <div key={i} className="p-4 bg-white rounded shadow">

//                 <h3 className="font-bold">{article.title}</h3>
//                 <p className="text-sm text-gray-500">{article.source}</p>

//                 <p className="mt-2">
//                   Sentiment:
//                   <span className={article.label === 'Bullish' ? 'text-green-500' : 'text-red-500'}>
//                     {" "}{article.label}
//                   </span>
//                 </p>

//                 <p>Confidence: {article.confidence}%</p>

//                 <div className="mt-2">
//                   <p className="font-semibold">Why?</p>
//                   <ul className="list-disc ml-5">
//                     {article.reasons.map((r, idx) => (
//                       <li key={idx}>{r}</li>
//                     ))}
//                   </ul>
//                 </div>

//                 <p className="mt-2 text-gray-600">{article.content}</p>
//               </div>
//             ))}
//           </div>
//         </>
//       )}

//     </div>
//   );
// };

// export default App;