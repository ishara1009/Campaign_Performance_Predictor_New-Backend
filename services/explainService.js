const axios = require('axios');

const GROQ_API = 'https://api.groq.com/openai/v1/chat/completions';

/**
 * Generates detailed explainability insights using Groq LLM.
 * @param {object} data – combined inputs + predictions
 */
async function generateInsights(data) {
  const {
    caption, content, platform, post_date, post_time,
    followers, ad_boost,
    likes, comments, shares, clicks, timing_quality_score,
  } = data;

  const systemPrompt = `You are an expert social media marketing analyst specialising in Sri Lankan SMEs. 
Your role is to provide actionable, research-grade explanations on how to improve social media post performance.
Always respond in structured JSON format as described.`;

  const userPrompt = `A social media post was analysed with the following details and AI predictions:

POST DETAILS:
- Platform: ${platform}
- Caption: "${caption || 'Not provided'}"
- Content: "${content || 'Not provided'}"
- Post Date: ${post_date}
- Post Time: ${post_time}
- Followers: ${followers}
- Ad Boost: ${ad_boost === 1 || ad_boost === '1' || ad_boost === true ? 'Yes' : 'No'}

PREDICTED PERFORMANCE:
- Likes: ${typeof likes === 'number' ? likes.toFixed(0) : likes}
- Comments: ${typeof comments === 'number' ? comments.toFixed(0) : comments}
- Shares: ${typeof shares === 'number' ? shares.toFixed(0) : shares}
- Clicks: ${typeof clicks === 'number' ? clicks.toFixed(0) : clicks}
- Timing Quality Score: ${typeof timing_quality_score === 'number' ? timing_quality_score.toFixed(2) : timing_quality_score} / 1.0

Based on these predictions, provide a comprehensive explainability report. Respond ONLY with a JSON object matching this exact schema:

{
  "overall_assessment": "2-3 sentence overall assessment of predicted performance",
  "performance_level": "Low | Moderate | Good | Excellent",
  "improvements": [
    {
      "metric": "Likes | Comments | Shares | Clicks | Timing Quality Score",
      "current_score": "current predicted value",
      "improvement_tips": ["tip 1", "tip 2", "tip 3"]
    }
  ],
  "caption_advice": "Specific advice to make the caption more engaging and attractive for the content",
  "hashtag_suggestions": ["#hashtag1", "#hashtag2", "#hashtag3", "#hashtag4", "#hashtag5", "#hashtag6", "#hashtag7", "#hashtag8"],
  "content_quality_tips": ["tip about image/content clarity", "another visual tip"],
  "best_posting_time": {
    "recommended_days": ["Monday", "Wednesday"],
    "recommended_hours": "e.g. 7 PM – 9 PM",
    "reasoning": "Why this time is optimal"
  },
  "platform_specific_tips": ["platform-specific tip 1", "platform-specific tip 2"],
  "ad_boost_advice": "Specific advice on whether to use ad boost and how",
  "novelty_insight": "One unique research insight about this post that could significantly increase engagement"
}`;

  const response = await axios.post(
    GROQ_API,
    {
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      temperature: 0.4,
      max_tokens: 2048,
      response_format: { type: 'json_object' },
    },
    {
      headers: {
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
        'Content-Type': 'application/json',
      },
    }
  );

  const rawContent = response.data?.choices?.[0]?.message?.content || '{}';

  let parsed;
  try {
    parsed = JSON.parse(rawContent);
  } catch {
    parsed = { overall_assessment: rawContent, performance_level: 'Moderate', improvements: [] };
  }

  return parsed;
}

module.exports = { generateInsights };
