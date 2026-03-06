const express = require('express');
const router = express.Router();
const { runShapLime } = require('../controllers/shapLimeController');

// POST /api/shap-lime
router.post('/', runShapLime);

module.exports = router;
