const supabase = require('../services/supabaseService');

/**
 * GET /api/history
 * Returns latest 50 predictions from Supabase
 */
async function getPredictions(req, res) {
  try {
    const { data, error } = await supabase
      .from('predictions')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(50);

    if (error) throw new Error(error.message);
    return res.json({ success: true, data });
  } catch (err) {
    console.error('History fetch error:', err.message);
    return res.status(500).json({ error: err.message });
  }
}

/**
 * DELETE /api/history/:id
 */
async function deletePrediction(req, res) {
  try {
    const { id } = req.params;
    const { error } = await supabase.from('predictions').delete().eq('id', id);
    if (error) throw new Error(error.message);
    return res.json({ success: true });
  } catch (err) {
    console.error('Delete error:', err.message);
    return res.status(500).json({ error: err.message });
  }
}

module.exports = { getPredictions, deletePrediction };
