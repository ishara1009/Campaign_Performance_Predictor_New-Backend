const fs = require('fs');
const axios = require('axios');

const token = process.env.VISION_TOKEN;
const VISION_API = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${token}`;

/**
 * Extracts text (including Sinhala / Sinhala+English) from an image file
 * using the vision language model.
 */
async function extractText(imagePath, mimeType) {
  const imageBytes = fs.readFileSync(imagePath);
  const base64Image = imageBytes.toString('base64');

  const requestBody = {
    contents: [
      {
        parts: [
          {
            text:
              'Please extract all text from this image exactly as it appears. ' +
              'The text may be in Sinhala, English, or a mix of both. ' +
              'Return all the extracted text as a single coherent paragraph — ' +
              'do NOT split into separate lines per word. Keep sentences together meaningfully. ' +
              'If no text is found, return: "No text detected."',
          },
          {
            inline_data: {
              mime_type: mimeType,
              data: base64Image,
            },
          },
        ],
      },
    ],
    generationConfig: {
      temperature: 0.1,
      maxOutputTokens: 1024,
    },
  };

  const response = await axios.post(VISION_API, requestBody, {
    headers: { 'Content-Type': 'application/json' },
  });

  const candidates = response.data?.candidates;
  if (!candidates || candidates.length === 0) {
    throw new Error('No response from vision service');
  }

  const rawText = candidates[0]?.content?.parts?.[0]?.text || '';
  // Normalise: collapse excessive newlines into spaces and trim
  return rawText
    .replace(/\r?\n/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

module.exports = { extractText };
