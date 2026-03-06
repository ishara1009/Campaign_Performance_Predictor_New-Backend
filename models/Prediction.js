const mongoose = require('mongoose');

const PredictionSchema = new mongoose.Schema(
  {
    // Form inputs
    caption:  { type: String, default: '' },
    content:  { type: String, default: '' },
    platform: { type: String, required: true },
    post_date: { type: String, required: true },
    post_time: { type: String, required: true },
    followers: { type: Number, required: true },
    ad_boost:  { type: Number, default: 0 },

    // Prediction outputs
    likes:                { type: Number },
    comments:             { type: Number },
    shares:               { type: Number },
    clicks:               { type: Number },
    timing_quality_score: { type: Number },

    // AI Explainability (saved after /api/explain is called)
    explanation: { type: mongoose.Schema.Types.Mixed, default: null },
  },
  { timestamps: true }
);

module.exports = mongoose.model('Prediction', PredictionSchema);
