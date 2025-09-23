@echo off
echo Starting CancerScope Backend...

REM Check if .env file exists
if not exist .env (
    echo Creating .env file from template...
    (
        echo PORT=3001
        echo NODE_ENV=development
        echo FRONTEND_URL=http://localhost:5173
        echo PYTHON_ML_SERVICE_URL=http://localhost:8000
    ) > .env
    echo .env file created!
)

REM Install dependencies if node_modules doesn't exist
if not exist node_modules (
    echo Installing dependencies...
    npm install
)

REM Create uploads directory if it doesn't exist
if not exist uploads (
    echo Creating uploads directory...
    mkdir uploads
)

REM Start the server
echo Starting server on port 3001...
npm start

