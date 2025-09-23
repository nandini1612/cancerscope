const express = require('express');
const { getMetricsController } = require('../controllers/metricsController');

const router = express.Router();

// GET /metrics - Get model performance metrics
router.get('/', getMetricsController);

module.exports = router;

