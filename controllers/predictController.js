const { spawn } = require('child_process');
const path = require('path');
const supabase = require('../services/supabaseService');

/**
 * POST /api/predict
 * Body: { caption, content, platform, post_date, post_time, followers, ad_boost }
 */
async function runPrediction(req, res) {
  try {
    const { caption, content, platform, post_date, post_time, followers, ad_boost } = req.body;

    // Validate required fields
    if (!platform || !post_date || !post_time || followers === undefined || ad_boost === undefined) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const inputPayload = {
      caption: caption || '',
      content: content || '',
      platform: platform,
      post_date: post_date,
      post_time: post_time,
      followers: Number(followers),
      ad_boost: Number(ad_boost),
    };

    // Call Python inference script
    const pythonPath = process.env.PYTHON_PATH || 'python';
    const scriptPath = path.join(__dirname, '../python/predict.py');
    const modelPath  = path.join(__dirname, '../SavedModels/Transformer.keras');

    const prediction = await callPythonPredict(pythonPath, scriptPath, modelPath, inputPayload);

    // Persist to Supabase
    const { data: savedRow, error: dbErr } = await supabase
      .from('predictions')
      .insert([
        {
          caption: inputPayload.caption,
          content: inputPayload.content,
          platform: inputPayload.platform,
          post_date: inputPayload.post_date,
          post_time: inputPayload.post_time,
          followers: inputPayload.followers,
          ad_boost: inputPayload.ad_boost,
          likes: prediction.likes,
          comments: prediction.comments,
          shares: prediction.shares,
          clicks: prediction.clicks,
          timing_quality_score: prediction.timing_quality_score,
          created_at: new Date().toISOString(),
        },
      ])
      .select()
      .single();

    if (dbErr) {
      console.error('Supabase insert error:', dbErr.message);
      // Return prediction even if DB save fails
    }

    return res.json({
      success: true,
      prediction,
      id: savedRow?.id || null,
    });
  } catch (err) {
    console.error('Prediction error:', err.message);
    return res.status(500).json({ error: err.message || 'Prediction failed' });
  }
}

function callPythonPredict(pythonPath, scriptPath, modelPath, payload) {
  return new Promise((resolve, reject) => {
    const args = [scriptPath, modelPath, JSON.stringify(payload)];
    const proc = spawn(pythonPath, args);

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (d) => (stdout += d.toString()));
    proc.stderr.on('data', (d) => (stderr += d.toString()));

    proc.on('close', (code) => {
      if (code !== 0) {
        return reject(new Error(`Python script exited with code ${code}: ${stderr}`));
      }
      try {
        const result = JSON.parse(stdout.trim());
        if (result.error) return reject(new Error(result.error));
        resolve(result);
      } catch {
        reject(new Error(`Failed to parse Python output: ${stdout}`));
      }
    });
  });
}

module.exports = { runPrediction };
