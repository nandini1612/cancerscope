// Set TensorFlow environment variable to reduce warnings
process.env.TF_CPP_MIN_LOG_LEVEL = '2';
process.env.TF_ENABLE_ONEDNN_OPTS = '0';

const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const axios = require('axios');
const FormData = require('form-data');
const { v4: uuidv4 } = require('uuid');

const app = express();
const PORT = 3001;
const PYTHON_SERVICE_URL = 'http://localhost:5000';

// CORS configuration
app.use(cors({
  origin: ['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:8080', 'http://127.0.0.1:8080'],
  credentials: true
}));

app.use(express.json());

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Create results directory for storing outputs
const resultsDir = path.join(__dirname, 'results');
const modelsDir = path.join(__dirname, 'models');
const shapPlotsDir = path.join(resultsDir, 'shap_plots');
const gradcamPlotsDir = path.join(resultsDir, 'gradcam_plots');

[resultsDir, modelsDir, shapPlotsDir, gradcamPlotsDir].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Multer configuration for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    const uniqueName = `${uuidv4()}-${file.originalname}`;
    cb(null, uniqueName);
  }
});

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB limit
  },
  fileFilter: (req, file, cb) => {
    if (req.route.path === '/predict-image') {
      // Accept images
      if (file.mimetype.startsWith('image/')) {
        cb(null, true);
      } else {
        cb(new Error('Only image files are allowed'), false);
      }
    } else if (req.route.path.includes('/predict-tabular')) {
      // Accept CSV files
      if (file.mimetype === 'text/csv' || file.originalname.endsWith('.csv')) {
        cb(null, true);
      } else {
        cb(new Error('Only CSV files are allowed'), false);
      }
    } else {
      cb(null, true);
    }
  }
});

// Store latest metrics (in production, you'd use a database)
let latestMetrics = {
  sensitivity: 0.92,
  specificity: 0.89,
  rocAuc: 0.95,
  prAuc: 0.91
};

// Store latest predictions for download
let latestPredictions = [];

// Helper function to check Python service health
async function checkPythonService() {
  try {
    const response = await axios.get(`${PYTHON_SERVICE_URL}/health`, { timeout: 5000 });
    return response.status === 200;
  } catch (error) {
    console.error('Python service health check failed:', error.message);
    return false;
  }
}

// Helper function to handle Python service requests with error handling
async function callPythonService(endpoint, data = null, method = 'GET') {
  try {
    const config = {
      method,
      url: `${PYTHON_SERVICE_URL}${endpoint}`,
      timeout: 120000, // 2 minutes timeout for model operations
    };

    if (data) {
      if (data instanceof FormData) {
        config.data = data;
        config.headers = data.getHeaders();
      } else {
        config.data = data;
        config.headers = { 'Content-Type': 'application/json' };
      }
    }

    const response = await axios(config);
    return response.data;
  } catch (error) {
    console.error(`Python service call failed for ${endpoint}:`, error.message);
    
    if (error.response) {
      throw new Error(`Python service error (${error.response.status}): ${error.response.data?.error || error.response.statusText}`);
    } else if (error.code === 'ECONNREFUSED') {
      throw new Error('Python service is not running. Please start the Python service on port 5000.');
    } else if (error.code === 'ETIMEDOUT') {
      throw new Error('Python service request timed out. The operation may be taking longer than expected.');
    } else {
      throw new Error(`Failed to communicate with Python service: ${error.message}`);
    }
  }
}

// Health check endpoint
app.get('/health', async (req, res) => {
  const pythonServiceHealthy = await checkPythonService();
  res.json({ 
    status: 'ok',
    pythonService: pythonServiceHealthy ? 'healthy' : 'unavailable',
    timestamp: new Date().toISOString()
  });
});

// Get model metrics
app.get('/metrics', async (req, res) => {
  try {
    // Get metrics from Python service
    const metrics = await callPythonService('/metrics');
    res.json(metrics);
  } catch (error) {
    console.error('Metrics error:', error);
    // Return default metrics if service is unavailable
    res.json({
      sensitivity: 0.85,
      specificity: 0.82,
      rocAuc: 0.90,
      prAuc: 0.87,
      error: 'Using default metrics - Python service unavailable'
    });
  }
});

// Image prediction endpoint with Grad-CAM
app.post('/predict-image', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file uploaded' });
    }

    console.log('Processing image:', req.file.filename);
    
    try {
      // Create form data for Python service
      const formData = new FormData();
      formData.append('file', fs.createReadStream(req.file.path), {
        filename: req.file.originalname,
        contentType: req.file.mimetype
      });

      // Add patient ID if provided
      if (req.body.patientId) {
        formData.append('patientId', req.body.patientId);
      }

      // Call Python service for image prediction
      const result = await callPythonService('/predict-image', formData, 'POST');
      
      // Clean up uploaded file
      fs.unlinkSync(req.file.path);
      
      // Check if prediction was successful
      if (result.error) {
        return res.status(500).json({ 
          error: 'Failed to process image', 
          details: result.error 
        });
      }
      
      // Return prediction result with Grad-CAM URL
      res.json({
        predictedClass: result.predictedClass,
        confidence: result.confidence,
        gradCamUrl: result.gradCamUrl || null,
        explanationText: result.explanationText || 'Grad-CAM visualization shows areas the model focused on for this prediction.',
        patientId: result.patientId
      });
      
    } catch (serviceError) {
      console.error('Python service error:', serviceError);
      
      // Clean up uploaded file
      if (fs.existsSync(req.file.path)) {
        fs.unlinkSync(req.file.path);
      }
      
      res.status(500).json({ 
        error: 'Failed to process image', 
        details: serviceError.message 
      });
    }
    
  } catch (error) {
    console.error('Image prediction error:', error);
    res.status(500).json({ 
      error: 'Internal server error', 
      details: error.message 
    });
  }
});

// Tabular data prediction endpoint with SHAP
app.post('/predict-tabular/tabular', upload.single('file'), async (req, res) => {
  console.log('=== TABULAR PREDICTION REQUEST ===');
  try {
    if (!req.file) {
      console.log('ERROR: No CSV file uploaded');
      return res.status(400).json({ error: 'No CSV file uploaded' });
    }

    console.log('Processing CSV:', req.file.filename);
    console.log('File path:', req.file.path);
    console.log('File size:', req.file.size);
    
    try {
      console.log('Calling Python service for tabular prediction...');
      
      // Create form data for Python service
      const formData = new FormData();
      formData.append('file', fs.createReadStream(req.file.path), {
        filename: req.file.originalname,
        contentType: 'text/csv'
      });
      formData.append('includeShap', 'true');

      // Call Python service for tabular prediction with SHAP
      const result = await callPythonService('/predict-tabular', formData, 'POST');
      console.log('Python service result received');
      
      // Clean up uploaded file
      fs.unlinkSync(req.file.path);
      console.log('Cleaned up uploaded file');
      
      // Check if prediction was successful
      if (result.error) {
        console.log('Python service returned error:', result.error);
        return res.status(500).json({ 
          error: 'Failed to process CSV file', 
          details: result.error 
        });
      }
      
      // Store predictions for download (enhanced with SHAP data)
      latestPredictions = result.predictions || result || [];
      console.log('Stored predictions for download:', latestPredictions.length, 'items');
      
      // Return predictions with SHAP explanations
      console.log('Sending successful response with SHAP data');
      res.json(latestPredictions);
      
    } catch (serviceError) {
      console.error('Python service error:', serviceError);
      
      // Clean up uploaded file
      if (fs.existsSync(req.file.path)) {
        fs.unlinkSync(req.file.path);
      }
      
      res.status(500).json({ 
        error: 'Failed to process CSV file', 
        details: serviceError.message 
      });
    }
    
  } catch (error) {
    console.error('Tabular prediction error:', error);
    res.status(500).json({ 
      error: 'Internal server error', 
      details: error.message 
    });
  }
});

// SHAP-specific tabular prediction endpoint  
app.post('/predict-tabular/shap', upload.single('file'), async (req, res) => {
  console.log('=== SHAP PREDICTION REQUEST ===');
  try {
    if (!req.file) {
      console.log('ERROR: No CSV file uploaded');
      return res.status(400).json({ error: 'No CSV file uploaded' });
    }

    console.log('Processing CSV with SHAP:', req.file.filename);
    
    try {
      console.log('Calling Python service with SHAP enabled...');
      
      // Create form data for Python service
      const formData = new FormData();
      formData.append('file', fs.createReadStream(req.file.path), {
        filename: req.file.originalname,
        contentType: 'text/csv'
      });
      formData.append('includeShap', 'true');
      formData.append('generatePlots', 'true');

      // Call Python service for SHAP analysis
      const result = await callPythonService('/predict-tabular/shap', formData, 'POST');
      console.log('Python service SHAP result received');
      
      // Clean up uploaded file
      fs.unlinkSync(req.file.path);
      
      if (result.error) {
        return res.status(500).json({ 
          error: 'Failed to process CSV file', 
          details: result.error 
        });
      }
      
      // Store predictions for download
      latestPredictions = result.predictions || result || [];
      
      // Return the full result structure
      res.json(result);
      
    } catch (serviceError) {
      console.error('Python service error:', serviceError);
      if (fs.existsSync(req.file.path)) {
        fs.unlinkSync(req.file.path);
      }
      
      res.status(500).json({ 
        error: 'Failed to process CSV file', 
        details: serviceError.message 
      });
    }
    
  } catch (error) {
    console.error('SHAP prediction error:', error);
    res.status(500).json({ 
      error: 'Internal server error', 
      details: error.message 
    });
  }
});

// Metrics endpoint specifically for tabular model
app.get('/predict-tabular/metrics', async (req, res) => {
  try {
    const result = await callPythonService('/predict-tabular/metrics');
    res.json(result);
  } catch (error) {
    console.error('Tabular metrics error:', error);
    res.json({
      sensitivity: 0.85,
      specificity: 0.82,
      rocAuc: 0.90,
      prAuc: 0.87,
      error: 'Using default metrics - Python service unavailable'
    });
  }
});

// Download predictions CSV (enhanced with SHAP data)
app.get('/predict-tabular/download', (req, res) => {
  try {
    if (latestPredictions.length === 0) {
      return res.status(404).json({ error: 'No predictions available for download' });
    }
    
    // Generate CSV content with SHAP information
    const csvHeader = 'Patient ID,Predicted Class,Probability,Top Risk Factors,SHAP Plot URL\n';
    const csvRows = latestPredictions.map(pred => {
      const topFactors = pred.topFeatures ? pred.topFeatures.join(';') : 'N/A';
      const shapUrl = pred.shapPlotUrl || 'N/A';
      return `${pred.patientId},${pred.predictedClass},${pred.probability},"${topFactors}",${shapUrl}`;
    }).join('\n');
    
    const csvContent = csvHeader + csvRows;
    
    // Set response headers for file download
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', 'attachment; filename="predictions_with_explanations.csv"');
    
    res.send(csvContent);
    
  } catch (error) {
    console.error('Download error:', error);
    res.status(500).json({ 
      error: 'Failed to generate predictions file', 
      details: error.message 
    });
  }
});

// Get individual SHAP explanation for a specific patient
app.get('/shap-explanation/:patientId', async (req, res) => {
  try {
    const { patientId } = req.params;
    
    // Try to get from Python service first
    try {
      const result = await callPythonService(`/shap-explanation/${patientId}`);
      res.json(result);
      return;
    } catch (serviceError) {
      console.log('Python service unavailable, falling back to local data');
    }
    
    // Fallback to local data
    const prediction = latestPredictions.find(p => p.patientId === patientId);
    
    if (!prediction) {
      return res.status(404).json({ error: 'Patient prediction not found' });
    }
    
    if (!prediction.shapPlotUrl) {
      return res.status(404).json({ error: 'SHAP explanation not available for this patient' });
    }
    
    res.json({
      patientId: prediction.patientId,
      shapPlotUrl: prediction.shapPlotUrl,
      topFeatures: prediction.topFeatures || [],
      featureImportances: prediction.featureImportances || {},
      explanationText: prediction.explanationText || 'SHAP values show how each feature contributed to the prediction.'
    });
    
  } catch (error) {
    console.error('SHAP explanation error:', error);
    res.status(500).json({ 
      error: 'Failed to retrieve SHAP explanation', 
      details: error.message 
    });
  }
});

// Training endpoints for development/testing
app.post('/train-risk-model', async (req, res) => {
  try {
    console.log('Training risk model...');
    const requestData = {
      epochs: req.body.epochs || 50,
      dataset: req.body.dataset || 'default'
    };
    
    const result = await callPythonService('/train-risk-model', requestData, 'POST');
    res.json(result);
  } catch (error) {
    console.error('Risk model training error:', error);
    res.status(500).json({ 
      error: 'Failed to train risk model', 
      details: error.message 
    });
  }
});

app.post('/train-image-model', upload.single('dataset'), async (req, res) => {
  try {
    const epochs = req.body.epochs || 5;
    const datasetType = req.body.datasetType || 'histopathology';
    
    console.log(`Training image model for ${epochs} epochs with ${datasetType} dataset...`);
    
    let requestData;
    
    if (req.file) {
      // If dataset file uploaded
      const formData = new FormData();
      formData.append('file', fs.createReadStream(req.file.path), {
        filename: req.file.originalname,
        contentType: req.file.mimetype
      });
      formData.append('epochs', epochs.toString());
      formData.append('datasetType', datasetType);
      
      requestData = formData;
    } else {
      // If using existing dataset
      requestData = {
        epochs,
        datasetType,
        datasetPath: req.body.datasetPath || '/path/to/dataset'
      };
    }
    
    const result = await callPythonService('/train-image-model', requestData, 'POST');
    
    // Clean up uploaded file if exists
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    
    res.json(result);
  } catch (error) {
    console.error('Image model training error:', error);
    
    // Clean up uploaded file if exists
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    
    res.status(500).json({ 
      error: 'Failed to train image model', 
      details: error.message 
    });
  }
});

// Batch explanation generation endpoint
app.post('/generate-batch-explanations', async (req, res) => {
  try {
    const { patientIds } = req.body;
    
    if (!patientIds || !Array.isArray(patientIds)) {
      return res.status(400).json({ error: 'Invalid patient IDs provided' });
    }
    
    console.log('Generating batch explanations for patients:', patientIds);
    
    // Call Python service to generate batch explanations
    const requestData = { patientIds };
    const result = await callPythonService('/generate-batch-explanations', requestData, 'POST');
    
    if (result.error) {
      return res.status(500).json({ 
        error: 'Failed to generate batch explanations', 
        details: result.error 
      });
    }
    
    res.json({
      message: 'Batch explanations generated successfully',
      explanations: result.explanations || [],
      totalGenerated: result.totalGenerated || 0
    });
    
  } catch (error) {
    console.error('Batch explanation error:', error);
    res.status(500).json({ 
      error: 'Failed to generate batch explanations', 
      details: error.message 
    });
  }
});

// Get explanation statistics
app.get('/explanation-stats', async (req, res) => {
  try {
    // Try to get from Python service first
    try {
      const result = await callPythonService('/explanation-stats');
      res.json(result);
      return;
    } catch (serviceError) {
      console.log('Python service unavailable, calculating from local data');
    }
    
    // Fallback to local calculation
    const totalPredictions = latestPredictions.length;
    const predictionsWithShap = latestPredictions.filter(p => p.shapPlotUrl).length;
    const malignantPredictions = latestPredictions.filter(p => p.predictedClass === 'Malignant').length;
    const benignPredictions = latestPredictions.filter(p => p.predictedClass === 'Benign').length;
    
    // Get most common risk factors
    const allFeatures = latestPredictions.flatMap(p => p.topFeatures || []);
    const featureCount = {};
    allFeatures.forEach(feature => {
      featureCount[feature] = (featureCount[feature] || 0) + 1;
    });
    
    const topRiskFactors = Object.entries(featureCount)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([feature, count]) => ({ feature, count }));
    
    res.json({
      totalPredictions,
      predictionsWithExplanations: predictionsWithShap,
      malignantPredictions,
      benignPredictions,
      topRiskFactors,
      explanationCoverage: totalPredictions > 0 ? (predictionsWithShap / totalPredictions * 100).toFixed(1) : 0
    });
    
  } catch (error) {
    console.error('Explanation stats error:', error);
    res.status(500).json({ 
      error: 'Failed to generate explanation statistics', 
      details: error.message 
    });
  }
});

// Serve static files (for SHAP plots, Grad-CAM images, etc.)
app.use('/results', express.static(resultsDir));
app.use('/results/shap_plots', express.static(shapPlotsDir));
app.use('/results/gradcam_plots', express.static(gradcamPlotsDir));

// Error handling middleware
app.use((error, req, res, next) => {
  console.error('Unhandled error:', error);
  
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'File too large (max 50MB)' });
    }
    return res.status(400).json({ error: `File upload error: ${error.message}` });
  }
  
  res.status(500).json({ 
    error: 'Internal server error', 
    details: error.message 
  });
});

// Handle 404 routes
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\nShutting down server...');
  
  // Clean up temporary files
  try {
    if (fs.existsSync(uploadsDir)) {
      const files = fs.readdirSync(uploadsDir);
      files.forEach(file => {
        fs.unlinkSync(path.join(uploadsDir, file));
      });
    }
  } catch (error) {
    console.error('Error cleaning up files:', error);
  }
  
  process.exit(0);
});

// Start server
app.listen(PORT, async () => {
  console.log(`🚀 CancerScope Backend running on http://localhost:${PORT}`);
  console.log(`🐍 Python Service URL: ${PYTHON_SERVICE_URL}`);
  console.log(`📁 Uploads directory: ${uploadsDir}`);
  console.log(`📊 Results directory: ${resultsDir}`);
  console.log(`🤖 Models directory: ${modelsDir}`);
  console.log(`📈 SHAP plots directory: ${shapPlotsDir}`);
  console.log(`🎯 Grad-CAM plots directory: ${gradcamPlotsDir}`);
  
  // Check Python service health on startup
  const pythonServiceHealthy = await checkPythonService();
  console.log(`🔗 Python Service Status: ${pythonServiceHealthy ? '✅ Healthy' : '❌ Unavailable'}`);
  
  if (!pythonServiceHealthy) {
    console.log(`⚠️  Warning: Python service is not running on ${PYTHON_SERVICE_URL}`);
    console.log(`   Please start the Python service before making predictions.`);
  }
  
  console.log('\n📋 Available endpoints:');
  console.log(`  GET  /health - Health check (includes Python service status)`);
  console.log(`  GET  /metrics - Model performance metrics`);
  console.log(`  POST /predict-image - Image prediction with Grad-CAM`);
  console.log(`  POST /predict-tabular/tabular - Tabular data prediction with SHAP`);
  console.log(`  POST /predict-tabular/shap - SHAP-specific tabular prediction`);
  console.log(`  GET  /predict-tabular/metrics - Tabular model metrics`);
  console.log(`  GET  /predict-tabular/download - Download predictions CSV with explanations`);
  console.log(`  GET  /shap-explanation/:patientId - Get individual SHAP explanation`);
  console.log(`  POST /generate-batch-explanations - Generate batch explanations`);
  console.log(`  GET  /explanation-stats - Get explanation statistics`);
  console.log(`  POST /train-risk-model - Train risk model (dev)`);
  console.log(`  POST /train-image-model - Train image model (dev)`);
  console.log(`  GET  /results/* - Serve static result files`);
  console.log(`  GET  /results/shap_plots/* - Serve SHAP plot images`);
  console.log(`  GET  /results/gradcam_plots/* - Serve Grad-CAM images\n`);
});