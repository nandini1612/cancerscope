# CancerScope Backend API Documentation

## Overview

The CancerScope Backend provides RESTful APIs for breast cancer prediction using machine learning models. The API accepts CSV files with patient diagnostic features and returns predictions with confidence scores.

## Base URL

```
http://localhost:3001
```

## Authentication

Currently, no authentication is required. In production, implement proper authentication mechanisms.

## Endpoints

### 1. Health Check

**GET** `/health`

Check if the API is running.

**Response:**
```json
{
  "status": "OK",
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

### 2. Upload CSV and Get Predictions

**POST** `/predict-tabular`

Upload a CSV file containing patient diagnostic features and receive predictions.

**Request:**
- **Content-Type:** `multipart/form-data`
- **Body:** 
  - `csvFile`: CSV file (required)

**CSV Format:**
The CSV should contain columns from the UCI Breast Cancer Dataset:
```csv
id,diagnosis,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean
842302,M,17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871
```

**Response:**
```json
[
  {
    "patientId": "842302",
    "predictedClass": "Malignant",
    "probability": 92.5,
    "shapPlotUrl": "https://example.com/shap/842302.png"
  },
  {
    "patientId": "842517",
    "predictedClass": "Benign",
    "probability": 98.1,
    "shapPlotUrl": "https://example.com/shap/842517.png"
  }
]
```

**Response Fields:**
- `patientId` (string): Unique identifier for the patient
- `predictedClass` (string): Either "Malignant" or "Benign"
- `probability` (number): Confidence score as percentage (0-100)
- `shapPlotUrl` (string): URL to SHAP explanation plot

**Error Responses:**
- `400 Bad Request`: No file uploaded or invalid CSV format
- `500 Internal Server Error`: Processing error

### 3. Download Predictions

**GET** `/download-tabular`

Download the most recent predictions as a CSV file.

**Response:**
- **Content-Type:** `text/csv`
- **Content-Disposition:** `attachment; filename="predictions.csv"`

**CSV Format:**
```csv
Patient ID,Predicted Class,Probability (%),SHAP Plot URL
842302,Malignant,92.5,https://example.com/shap/842302.png
842517,Benign,98.1,https://example.com/shap/842517.png
```

**Error Responses:**
- `404 Not Found`: No predictions available for download

### 4. Get Model Metrics

**GET** `/metrics`

Retrieve model performance metrics.

**Response:**
```json
{
  "sensitivity": 0.95,
  "specificity": 0.92,
  "rocAuc": 0.97,
  "prAuc": 0.94
}
```

**Response Fields:**
- `sensitivity` (number): True positive rate (0-1)
- `specificity` (number): True negative rate (0-1)
- `rocAuc` (number): ROC Area Under Curve (0-1)
- `prAuc` (number): Precision-Recall Area Under Curve (0-1)

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": "Error type",
  "message": "Detailed error message"
}
```

Common HTTP status codes:
- `200 OK`: Success
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

## Python ML Service Integration

The backend integrates with a Python ML service for actual predictions. If the Python service is unavailable, mock data is returned for development purposes.

**Expected Python Service Endpoints:**
- `POST /predict`: Accept CSV file and return predictions
- `GET /metrics`: Return model performance metrics

**Python Service Response Format:**
```json
{
  "predictions": [
    {
      "predicted_class": 1,
      "probability": 0.925,
      "shap_plot_url": "https://example.com/shap/842302.png"
    }
  ]
}
```

## Rate Limiting

Currently, no rate limiting is implemented. In production, implement rate limiting to prevent abuse.

## File Upload Limits

- Maximum file size: 10MB
- Allowed file types: CSV only
- File validation: Checks for CSV format and required columns

## CORS Configuration

The API is configured to accept requests from:
- Development: `http://localhost:5173`
- Production: Configure via `FRONTEND_URL` environment variable

## Security Features

- Helmet.js for security headers
- File upload validation
- Input sanitization
- Request size limits
- CORS protection

## Development vs Production

**Development Mode:**
- Detailed error messages
- Mock data fallback when Python service unavailable
- Verbose logging

**Production Mode:**
- Generic error messages
- Enhanced security
- Optimized performance

