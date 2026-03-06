const { spawn } = require('child_process');
const path = require('path');

/**
 * POST /api/shap-lime
 * Body: { caption, content, platform, post_date, post_time, followers, ad_boost }
 *
 * Runs the Python explain_shap_lime.py script to compute:
 *  - SHAP KernelExplainer values for all numeric features × all 5 output targets
 *  - LIME TabularExplainer importances for all numeric features × all 5 targets
 *  - LIME TextExplainer word-level attributions for caption + content × all 5 targets
 *  - Plain-English summaries and SHAP/LIME concordance check
 *
 * Expected response time: 30–90 seconds depending on hardware.
 * The frontend should display a dedicated loading state for this endpoint.
 */
async function runShapLime(req, res) {
  try {
    const { caption, content, platform, post_date, post_time, followers, ad_boost } = req.body;

    if (!platform || !post_date || !post_time || followers === undefined) {
      return res.status(400).json({ error: 'Missing required fields: platform, post_date, post_time, followers' });
    }

    const payload = {
      caption:   caption   || '',
      content:   content   || '',
      platform:  String(platform),
      post_date: String(post_date),
      post_time: String(post_time),
      followers: Number(followers) || 1000,
      ad_boost:  Number(ad_boost)  || 0,
    };

    const pythonPath = process.env.PYTHON_PATH || 'python';
    const scriptPath = path.join(__dirname, '../python/explain_shap_lime.py');
    const modelPath  = path.join(__dirname, '../SavedModels/Transformer.keras');

    console.log(`[shapLime] Starting SHAP/LIME analysis for platform=${payload.platform} followers=${payload.followers}`);

    const result = await callPython(pythonPath, scriptPath, modelPath, payload);

    console.log('[shapLime] Analysis complete');
    return res.json({ success: true, shapLime: result });
  } catch (err) {
    console.error('[shapLime] Error:', err.message);
    return res.status(500).json({ error: err.message || 'SHAP/LIME analysis failed' });
  }
}

/**
 * Spawns the Python explain_shap_lime.py script and collects its JSON output.
 * Times out after 180 seconds — SHAP KernelExplainer can be slow on large models.
 */
function callPython(pythonPath, scriptPath, modelPath, payload) {
  return new Promise((resolve, reject) => {
    const args = [scriptPath, modelPath, JSON.stringify(payload)];
    const proc = spawn(pythonPath, args, { env: { ...process.env } });

    let stdout = '';
    let stderr = '';
    let timedOut = false;

    // 180-second hard timeout
    const timer = setTimeout(() => {
      timedOut = true;
      proc.kill('SIGTERM');
      reject(new Error('SHAP/LIME analysis timed out after 180 seconds'));
    }, 180_000);

    proc.stdout.on('data', (d) => (stdout += d.toString()));
    proc.stderr.on('data', (d) => (stderr += d.toString()));

    proc.on('close', (code) => {
      clearTimeout(timer);
      if (timedOut) return;  // already rejected

      if (code !== 0) {
        const msg = stderr.slice(0, 800) || 'Script returned non-zero exit code';
        return reject(new Error(`SHAP/LIME script exited with code ${code}: ${msg}`));
      }

      const raw = stdout.trim();
      if (!raw) {
        return reject(new Error('SHAP/LIME script produced no output'));
      }

      try {
        const result = JSON.parse(raw);
        if (result.error) {
          return reject(new Error(result.error));
        }
        resolve(result);
      } catch {
        reject(new Error(`Failed to parse SHAP/LIME output: ${raw.slice(0, 300)}`));
      }
    });

    proc.on('error', (err) => {
      clearTimeout(timer);
      reject(new Error(`Failed to spawn Python process: ${err.message}`));
    });
  });
}

module.exports = { runShapLime };
