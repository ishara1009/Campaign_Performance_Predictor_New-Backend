const express = require('express');
const router = express.Router();
const { getPredictions, deletePrediction } = require('../controllers/historyController');

// GET  /api/history
router.get('/', getPredictions);

// DELETE /api/history/:id
router.delete('/:id', deletePrediction);

module.exports = router;
