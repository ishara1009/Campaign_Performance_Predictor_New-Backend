const fs = require('fs');
const axios = require('axios');

// Use environment variable with the provided Gemini API key as fallback
const GEMINI_API_KEY = process.env.VISION_TOKEN || process.env.GEMINI_API_KEY || 'AIzaSyDq4OganKVo1zTRFaNu-Xx-v7ONYpTTihQ';

// Use gemini-2.0-flash via the v1beta endpoint for best vision/OCR results
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`;

/**
 * Extracts text (including Sinhala / Sinhala+English) from an image file
 * using the Gemini 2.0 Flash vision model.
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
              'You are an expert OCR assistant. Carefully extract ALL visible text from this image exactly as it appears. ' +
              'The text may be in Sinhala, English, or a mix of both languages. ' +
              'Rules:\n' +
              '1. Extract every word, number, symbol, and punctuation visible in the image.\n' +
              '2. Preserve the original reading order (left to right, top to bottom).\n' +
              '3. Return the extracted text as a single coherent paragraph.\n' +
              '4. Do NOT add commentary, explanations, or formatting — only the raw extracted text.\n' +
              '5. If absolutely no text is visible in the image, return exactly: "No text detected."\n',
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
      temperature: 0.0,
      maxOutputTokens: 2048,
    },
  };

  let response;
  try {
    response = await axios.post(GEMINI_API_URL, requestBody, {
      headers: { 'Content-Type': 'application/json' },
      timeout: 30000,
    });
  } catch (axiosErr) {
    const status = axiosErr.response?.status;
    const detail = axiosErr.response?.data?.error?.message || axiosErr.message;
    throw new Error(`Gemini API error (${status || 'network'}): ${detail}`);
  }

  const candidates = response.data?.candidates;
  if (!candidates || candidates.length === 0) {
    throw new Error('No response from Gemini vision service');
  }

  const rawText = candidates[0]?.content?.parts?.[0]?.text || '';

  if (!rawText.trim()) {
    return 'No text detected.';
  }

  // Normalise: collapse excessive newlines into spaces and trim
  return rawText
    .replace(/\r?\n+/g, ' ')
    .replace(/\s{2,}/g, ' ')
    .trim();
}

module.exports = { extractText };
