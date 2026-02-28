const { extractText } = require('../services/textExtractorService');
const fs = require('fs');

/**
 * POST /api/extract
 * multipart/form-data: field "image"
 */
async function extractTextFromImage(req, res) {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file uploaded' });
    }

    const imagePath = req.file.path;
    const mimeType  = req.file.mimetype;

    const extractedText = await extractText(imagePath, mimeType);

    // Clean up uploaded file
    fs.unlink(imagePath, () => {});

    return res.json({ success: true, text: extractedText });
  } catch (err) {
    console.error('Text extraction error:', err.message);
    return res.status(500).json({ error: err.message || 'Text extraction failed' });
  }
}

module.exports = { extractTextFromImage };
