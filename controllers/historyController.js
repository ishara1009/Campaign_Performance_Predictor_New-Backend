const Prediction = require('../models/Prediction');

/**
 * GET /api/history
 * Returns latest 50 predictions from MongoDB
 */
async function getPredictions(req, res) {
  try {
    const data = await Prediction.find()
      .sort({ createdAt: -1 })
      .limit(50)
      .lean();
    return res.json({ success: true, data });
  } catch (err) {
    console.error('History fetch error:', err.message);
    return res.status(500).json({ error: err.message });
  }
}

/**
 * GET /api/history/:id
 * Returns a single prediction by MongoDB _id
 */
async function getPredictionById(req, res) {
  try {
    const doc = await Prediction.findById(req.params.id).lean();
    if (!doc) return res.status(404).json({ error: 'Prediction not found' });
    return res.json({ success: true, data: doc });
  } catch (err) {
    console.error('History getById error:', err.message);
    return res.status(500).json({ error: err.message });
  }
}

/**
 * DELETE /api/history/:id
 */
async function deletePrediction(req, res) {
  try {
    const doc = await Prediction.findByIdAndDelete(req.params.id);
    if (!doc) return res.status(404).json({ error: 'Prediction not found' });
    return res.json({ success: true });
  } catch (err) {
    console.error('Delete error:', err.message);
    return res.status(500).json({ error: err.message });
  }
}

module.exports = { getPredictions, getPredictionById, deletePrediction };

