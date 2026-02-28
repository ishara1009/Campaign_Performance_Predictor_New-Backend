const express = require('express');
const router = express.Router();
const { runPrediction } = require('../controllers/predictController');

// POST /api/predict
router.post('/', runPrediction);

module.exports = router;
