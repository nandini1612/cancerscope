#!/bin/bash

# CancerScope Backend Startup Script

echo "Starting CancerScope Backend..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cat > .env << EOF
PORT=3001
NODE_ENV=development
FRONTEND_URL=http://localhost:5173
PYTHON_ML_SERVICE_URL=http://localhost:8000
EOF
    echo ".env file created!"
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Create uploads directory if it doesn't exist
if [ ! -d "uploads" ]; then
    echo "Creating uploads directory..."
    mkdir -p uploads
fi

# Start the server
echo "Starting server on port 3001..."
npm start

