import { useState, useEffect, useCallback } from 'react';
import {
  fetchStockData,
  analyzeSentiment,
  fetchPortfolioData,
  fetchMarketData,
  fetchSectorAnalysis,
  fetchGeneralNews,
  fetchRecommendations
} from '../services/api';

export const useApiData = () => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [portfolioData, setPortfolioData] = useState<any>(null);
  const [newsData, setNewsData] = useState<any[]>([]);
  const [marketData, setMarketData] = useState<any>(null);
  const [sectorData, setSectorData] = useState<any>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);

    // Use allSettled so one failing endpoint (e.g. /dashboard) does not block news/sectors
    const results = await Promise.allSettled([
      fetchGeneralNews(),
      fetchMarketData(),
      fetchSectorAnalysis(),
    ]);

    const news = results[0].status === 'fulfilled' ? results[0].value : [];
    const market = results[1].status === 'fulfilled' ? results[1].value : null;
    const sectors =
      results[2].status === 'fulfilled'
        ? results[2].value
        : {
            market_trend: 'Neutral',
            spy_performance: 0,
            market_outlook: 'Market data unavailable',
            top_sectors: [],
            timestamp: new Date().toISOString(),
          };

    setNewsData(Array.isArray(news) ? news : []);
    setMarketData(market);
    setSectorData(sectors);

    results.forEach((r, i) => {
      if (r.status === 'rejected') {
        console.error(`fetchData source ${i} failed:`, r.reason);
      }
    });

    setLoading(false);
  }, []);

  const uploadPortfolio = useCallback(async (file: File) => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchPortfolioData(file);
      setPortfolioData(data);
      return data;
    } catch (err) {
      setError('Failed to upload portfolio');
      console.error('Error uploading portfolio:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const getStockData = useCallback(async (symbol: string) => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchStockData(symbol);
      return data;
    } catch (err) {
      setError('Failed to fetch stock data');
      console.error('Error fetching stock data:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const getSentimentAnalysis = useCallback(async (text: string) => {
    try {
      setLoading(true);
      setError(null);
      const data = await analyzeSentiment(text);
      return data;
    } catch (err) {
      setError('Failed to analyze sentiment');
      console.error('Error analyzing sentiment:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // Add function to fetch general news
  const getGeneralNews = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchGeneralNews();
      return data;
    } catch (err) {
      setError('Failed to fetch general news');
      console.error('Error fetching general news:', err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  // Add function to fetch recommendations
  const getRecommendations = useCallback(async () => {
    try {
      console.log('Fetching recommendations...');
      const data = await fetchRecommendations();
      console.log('Recommendations received:', data);
      return data;
    } catch (e) {
      console.error('Error fetching recommendations:', e);
      return [];
    }
  }, []);

  // Initial data fetch
  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Set up real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      fetchData();
    }, 60000); // Update every minute

    return () => clearInterval(interval);
  }, [fetchData]);

  return {
    loading,
    error,
    portfolioData,
    newsData,
    marketData,
    sectorData,
    getStockData,
    getSentimentAnalysis,
    uploadPortfolio,
    refreshData: fetchData,
    getGeneralNews,
    getRecommendations
  };
}; 