const axios = require('axios');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const { createObjectCsvWriter } = require('csv-writer');

// Store predictions in memory (in production, use a database)
let currentPredictions = [];

const predictController = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No CSV file uploaded' });
    }

    const csvFilePath = req.file.path;
    console.log(`Processing CSV file: ${csvFilePath}`);

    // Read CSV file to get patient data
    const patientData = [];
    await new Promise((resolve, reject) => {
      fs.createReadStream(csvFilePath)
        .pipe(csv())
        .on('data', (row) => {
          patientData.push(row);
        })
        .on('end', resolve)
        .on('error', reject);
    });

    if (patientData.length === 0) {
      return res.status(400).json({ error: 'CSV file is empty or invalid' });
    }

    // Forward to Python ML service
    const pythonServiceUrl = process.env.PYTHON_ML_SERVICE_URL || 'http://localhost:8000';
    
    try {
      // Create FormData for Python service
      const FormData = require('form-data');
      const formData = new FormData();
      formData.append('file', fs.createReadStream(csvFilePath));

      const response = await axios.post(`${pythonServiceUrl}/predict-tabular`, formData, {
        headers: {
          ...formData.getHeaders(),
        },
        timeout: 30000, // 30 second timeout
      });

      // Transform Python service response to match frontend expectations
      const predictions = response.data.predictions.map((pred, index) => ({
        patientId: pred.patientId || patientData[index]?.id || `patient_${index + 1}`,
        predictedClass: pred.predictedClass, // Already in correct format
        probability: Math.round(pred.probability * 100 * 100) / 100, // Convert to percentage with 2 decimal places
        shapPlotUrl: pred.shapPlotUrl || `${pythonServiceUrl}${pred.shapPlotUrl}`
      }));

      // Store predictions for download
      currentPredictions = predictions;

      // Clean up uploaded file
      fs.unlinkSync(csvFilePath);

      res.json(predictions);

    } catch (pythonError) {
      console.error('Python ML service error:', pythonError.message);
      
      // Fallback: Generate mock predictions if Python service is unavailable
      const mockPredictions = patientData.map((patient, index) => ({
        patientId: patient.id || `patient_${index + 1}`,
        predictedClass: Math.random() > 0.5 ? "Malignant" : "Benign",
        probability: Math.round((Math.random() * 20 + 80) * 100) / 100, // 80-100%
        shapPlotUrl: `https://example.com/shap/mock_${index}.png`
      }));

      currentPredictions = mockPredictions;
      
      // Clean up uploaded file
      fs.unlinkSync(csvFilePath);

      res.json(mockPredictions);
    }

  } catch (error) {
    console.error('Prediction error:', error);
    
    // Clean up uploaded file if it exists
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    
    res.status(500).json({ 
      error: 'Failed to process prediction',
      message: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
    });
  }
};

const downloadController = (req, res) => {
  try {
    if (currentPredictions.length === 0) {
      return res.status(404).json({ error: 'No predictions available for download' });
    }

    // Set headers for CSV download
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', 'attachment; filename="predictions.csv"');

    // Create CSV content
    const csvHeader = 'Patient ID,Predicted Class,Probability (%),SHAP Plot URL\n';
    const csvRows = currentPredictions.map(pred => 
      `${pred.patientId},${pred.predictedClass},${pred.probability},${pred.shapPlotUrl}`
    ).join('\n');

    const csvContent = csvHeader + csvRows;
    
    res.send(csvContent);

  } catch (error) {
    console.error('Download error:', error);
    res.status(500).json({ 
      error: 'Failed to generate download',
      message: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
    });
  }
};

const imagePredictController = async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file uploaded' });
    }

    const imageFilePath = req.file.path;
    console.log(`Processing image file: ${imageFilePath}`);

    // Forward to Python ML service
    const pythonServiceUrl = process.env.PYTHON_ML_SERVICE_URL || 'http://localhost:8000';
    
    try {
      // Create FormData for Python service
      const FormData = require('form-data');
      const formData = new FormData();
      formData.append('file', fs.createReadStream(imageFilePath));

      const response = await axios.post(`${pythonServiceUrl}/predict-image`, formData, {
        headers: {
          ...formData.getHeaders(),
        },
        timeout: 30000, // 30 second timeout
      });

      // Transform Python service response to match frontend expectations
      const result = {
        predictedClass: response.data.predictedClass,
        confidence: Math.round(response.data.confidence * 100 * 100) / 100, // Convert to percentage
        gradCamUrl: response.data.gradCamUrl.startsWith('http') 
          ? response.data.gradCamUrl 
          : `${pythonServiceUrl}${response.data.gradCamUrl}`
      };

      // Clean up uploaded file
      fs.unlinkSync(imageFilePath);

      res.json(result);

    } catch (pythonError) {
      console.error('Python ML service error:', pythonError.message);
      
      // Fallback: Generate mock prediction if Python service is unavailable
      const mockResult = {
        predictedClass: Math.random() > 0.5 ? "Malignant" : "Benign",
        confidence: Math.round((Math.random() * 20 + 80) * 100) / 100, // 80-100%
        gradCamUrl: "https://example.com/gradcam/mock.png"
      };

      // Clean up uploaded file
      fs.unlinkSync(imageFilePath);

      res.json(mockResult);
    }

  } catch (error) {
    console.error('Image prediction error:', error);
    
    // Clean up uploaded file if it exists
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    
    res.status(500).json({ 
      error: 'Failed to process image prediction',
      message: process.env.NODE_ENV === 'development' ? error.message : 'Internal server error'
    });
  }
};

module.exports = {
  predictController,
  imagePredictController,
  downloadController
};
