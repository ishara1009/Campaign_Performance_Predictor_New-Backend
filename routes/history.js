const express = require('express');
const router = express.Router();
const { getPredictions, getPredictionById, deletePrediction } = require('../controllers/historyController');

// GET  /api/history
router.get('/', getPredictions);

// GET  /api/history/:id
router.get('/:id', getPredictionById);

// DELETE /api/history/:id
router.delete('/:id', deletePrediction);

module.exports = router;
