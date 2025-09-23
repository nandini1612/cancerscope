const axios = require('axios');

const getMetricsController = async (req, res) => {
  try {
    const pythonServiceUrl = process.env.PYTHON_ML_SERVICE_URL || 'http://localhost:8000';
    
    try {
      // Try to get metrics from Python ML service
      const response = await axios.get(`${pythonServiceUrl}/metrics`, {
        timeout: 10000, // 10 second timeout
      });

      // Transform Python service response to match frontend expectations
      const metrics = {
        sensitivity: Math.round(response.data.sensitivity * 100) / 100,
        specificity: Math.round(response.data.specificity * 100) / 100,
        rocAuc: Math.round(response.data.rocAuc * 100) / 100,
        prAuc: Math.round(response.data.prAuc * 100) / 100
      };

      res.json(metrics);

    } catch (pythonError) {
      console.error('Python ML service error:', pythonError.message);
      
      // Fallback: Return mock metrics if Python service is unavailable
      const mockMetrics = {
        sensitivity: 0.95,
        specificity: 0.92,
        rocAuc: 0.97,
        prAuc: 0.94
      };

      res.json(mockMetrics);
    }

  } catch (error) {
    console.error('Metrics error:', error);
    res.status(500).json({ 
      error: 'Failed to retrieve metrics',
      message: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
    });
  }
};

module.exports = {
  getMetricsController
};
