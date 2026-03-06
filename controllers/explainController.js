const { generateInsights } = require('../services/explainService');
const Prediction = require('../models/Prediction');

/**
 * POST /api/explain
 * Body: { predictionId, caption, content, platform, post_date, post_time,
 *         followers, ad_boost, likes, comments, shares, clicks, timing_quality_score }
 */
async function generateExplanation(req, res) {
  try {
    const { predictionId, ...inputs } = req.body;

    if (!inputs.platform || inputs.likes === undefined) {
      return res.status(400).json({ error: 'Missing required prediction data' });
    }

    const explanation = await generateInsights(inputs);

    // Save explanation back to the MongoDB prediction document
    if (predictionId) {
      try {
        await Prediction.findByIdAndUpdate(predictionId, { explanation });
      } catch (dbErr) {
        console.error('MongoDB explanation update error:', dbErr.message);
      }
    }

    return res.json({ success: true, explanation });
  } catch (err) {
    console.error('Explanation error:', err.message);
    return res.status(500).json({ error: err.message || 'Explanation generation failed' });
  }
}

module.exports = { generateExplanation };
