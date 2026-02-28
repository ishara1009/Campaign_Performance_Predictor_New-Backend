const express = require('express');
const router = express.Router();
const { generateExplanation } = require('../controllers/explainController');

// POST /api/explain
router.post('/', generateExplanation);

module.exports = router;
