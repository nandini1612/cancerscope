/**
 * Utility functions for the CancerScope backend
 */

/**
 * Validates CSV file structure for breast cancer prediction
 * @param {Array} data - Parsed CSV data
 * @returns {Object} - Validation result with isValid and errors
 */
const validateCSVStructure = (data) => {
  const errors = [];
  
  if (!data || data.length === 0) {
    errors.push('CSV file is empty');
    return { isValid: false, errors };
  }

  // Check for required columns (based on UCI Breast Cancer Dataset)
  const requiredColumns = [
    'id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
    'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean'
  ];

  const firstRow = data[0];
  const missingColumns = requiredColumns.filter(col => !(col in firstRow));
  
  if (missingColumns.length > 0) {
    errors.push(`Missing required columns: ${missingColumns.join(', ')}`);
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};

/**
 * Sanitizes patient ID to prevent injection attacks
 * @param {string} patientId - Raw patient ID
 * @returns {string} - Sanitized patient ID
 */
const sanitizePatientId = (patientId) => {
  if (!patientId || typeof patientId !== 'string') {
    return 'unknown';
  }
  
  // Remove any potentially dangerous characters
  return patientId.replace(/[^a-zA-Z0-9_-]/g, '').substring(0, 50);
};

/**
 * Formats probability for display
 * @param {number} probability - Raw probability (0-1)
 * @returns {number} - Formatted probability percentage
 */
const formatProbability = (probability) => {
  if (typeof probability !== 'number' || isNaN(probability)) {
    return 0;
  }
  
  // Ensure probability is between 0 and 1
  const clamped = Math.max(0, Math.min(1, probability));
  
  // Convert to percentage and round to 2 decimal places
  return Math.round(clamped * 100 * 100) / 100;
};

/**
 * Generates a mock SHAP plot URL for demonstration
 * @param {string} patientId - Patient ID
 * @returns {string} - Mock SHAP plot URL
 */
const generateShapPlotUrl = (patientId) => {
  const sanitizedId = sanitizePatientId(patientId);
  return `https://example.com/shap/${sanitizedId}.png`;
};

module.exports = {
  validateCSVStructure,
  sanitizePatientId,
  formatProbability,
  generateShapPlotUrl
};

