# CancerScope Backend

Backend API for CancerScope - Minimal Breast Cancer Detection Explorer.

## Features

- **CSV Upload & Prediction**: Accept CSV files with patient diagnostic features and return predictions
- **Model Metrics**: Provide model performance metrics (sensitivity, specificity, ROC-AUC, PR-AUC)
- **CSV Download**: Download prediction results as CSV
- **Python ML Integration**: Forwards requests to Python ML service
- **CORS Support**: Configured for frontend integration
- **Error Handling**: Comprehensive error handling with fallback responses

## API Endpoints

### POST /predict-tabular
Upload CSV file and get predictions.

**Request:**
- Content-Type: `multipart/form-data`
- Body: CSV file with key `csvFile`

**Response:**
```json
[
  {
    "patientId": "842302",
    "predictedClass": "Malignant",
    "probability": 92.5,
    "shapPlotUrl": "https://example.com/shap/842302.png"
  }
]
```

### GET /download-tabular
Download predictions as CSV file.

**Response:**
- Content-Type: `text/csv`
- File: `predictions.csv`

### GET /metrics
Get model performance metrics.

**Response:**
```json
{
  "sensitivity": 0.95,
  "specificity": 0.92,
  "rocAuc": 0.97,
  "prAuc": 0.94
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "OK",
  "timestamp": "2024-01-01T00:00:00.000Z"
}
```

## Environment Variables

Create a `.env` file in the backend directory:

```env
PORT=3001
NODE_ENV=development
FRONTEND_URL=http://localhost:5173
PYTHON_ML_SERVICE_URL=http://localhost:8000
```

## Installation

```bash
cd backend
npm install
```

## Development

```bash
npm run dev
```

## Production

```bash
npm start
```

## Docker

```bash
docker build -t cancerscope-backend .
docker run -p 3001:3001 cancerscope-backend
```

## Python ML Service Integration

The backend expects a Python ML service running on `PYTHON_ML_SERVICE_URL` with the following endpoints:

- `POST /predict` - Accept CSV file and return predictions
- `GET /metrics` - Return model metrics

If the Python service is unavailable, the backend will return mock data for development purposes.

## CSV Format

Expected CSV format based on UCI Breast Cancer Dataset:

```csv
id,diagnosis,radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean
842302,M,17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871
```

## Security Features

- Helmet.js for security headers
- CORS configuration
- File upload validation
- Input sanitization
- Non-root Docker user
- Request size limits

