const axios = require('axios');
const {
  detectCategory,
  getBestPostingWindow,
  analyseText,
  analysePostTiming,
  AD_BOOST_IMPACT,
  DATASET_AVERAGES,
  HASHTAG_BENCHMARKS,
  OPTIMAL_HASHTAG_RANGE,
  OPTIMAL_CONTENT_LENGTH,
  BEST_DAYS,
  BEST_HOURS,
} = require('./datasetInsights');

const GROQ_API = 'https://api.groq.com/openai/v1/chat/completions';


/**
 * Generates detailed explainability insights using Groq LLM,
 * grounded in real statistics from the 125,000-row Sri Lankan SME dataset.
 * @param {object} data – combined inputs + predictions
 */
async function generateInsights(data) {
  const {
    caption, content, platform, post_date, post_time,
    followers, ad_boost,
    likes, comments, shares, clicks, timing_quality_score,
  } = data;

  // ── Pre-compute data-backed context ──────────────────────────────────────
  const category       = detectCategory(caption, content);
  const textStats      = analyseText(caption, content);
  const timingAnalysis = analysePostTiming(platform, post_date, post_time);
  const bestWindow     = getBestPostingWindow(platform, category);
  const adBoostLabel   = (ad_boost === 1 || ad_boost === '1' || ad_boost === true) ? 'Yes' : 'No';
  const isAdBoosted    = adBoostLabel === 'Yes';

  // ── Benchmark comparisons ────────────────────────────────────────────────
  const tqsPercent   = typeof timing_quality_score === 'number'
    ? Math.round(timing_quality_score * 100) : 0;
  const likesVal     = typeof likes    === 'number' ? Math.round(likes)    : 0;
  const commentsVal  = typeof comments === 'number' ? Math.round(comments) : 0;
  const sharesVal    = typeof shares   === 'number' ? Math.round(shares)   : 0;
  const clicksVal    = typeof clicks   === 'number' ? Math.round(clicks)   : 0;

  const vsAvgLikes    = likesVal   > DATASET_AVERAGES.likes    ? 'above' : 'below';
  const vsAvgComments = commentsVal> DATASET_AVERAGES.comments ? 'above' : 'below';
  const vsAvgClicks   = clicksVal  > DATASET_AVERAGES.clicks   ? 'above' : 'below';

  // Hashtag gap: what could be achieved with 7-9 hashtags
  const hashtagPotential = HASHTAG_BENCHMARKS['7-9'];

  // ── Unique post fingerprint (prevents identical responses for diff captions) ──
  const captionWords = (caption || '').trim().split(/\s+/).filter(Boolean);
  const fingerprint  = [
    captionWords.slice(0, 5).join(' ') || 'no caption',
    category,
    platform,
    `${timingAnalysis.dayName}-${timingAnalysis.hour}h`,
    `tqs${tqsPercent}`,
    `likes${likesVal}`,
    `ht${textStats.hashtagCount}`,
    isAdBoosted ? 'boosted' : 'organic',
  ].join('|');

  // ── System prompt ─────────────────────────────────────────────────────────
  const systemPrompt = `You are a senior data-driven social media strategist for Sri Lankan SMEs.
You have access to a 125,000-row dataset of real Sri Lankan social media posts.

YOUR ABSOLUTE RULES:
1. Every single insight MUST refer to specific words, phrases, or signals from the EXACT caption and content provided. Quote them directly.
2. The best_posting_time MUST use EXACTLY the recommended_days and recommended_hours injected in the prompt — do NOT change them.
3. All improvement tips must be actionable and specific to "${category}" businesses in Sri Lanka.
4. Compare the predicted metrics to the dataset benchmarks provided — say whether this post is above/below average and by how much.
5. Hashtag/CTA/emoji gaps must reference what is actually MISSING from the caption.
6. NEVER produce generic advice like "use engaging visuals" without tying it to the specific content.
7. The novelty_insight must be a unique, non-obvious strategy specific to this exact post.
8. Respond ONLY with valid JSON — no markdown, no extra text.`;

  // ── User prompt with dataset facts injected ───────────────────────────────
  const userPrompt = `POST FINGERPRINT (unique ID): ${fingerprint}

━━━ EXACT POST DETAILS ━━━
Platform  : ${platform}
Category  : ${category}
Caption   : "${caption || 'Not provided'}"
Content   : "${content || 'Not provided'}"
Post Date : ${post_date} (${timingAnalysis.dayName})
Post Time : ${post_time} (Hour ${timingAnalysis.hour}:00)
Followers : ${followers}
Ad Boost  : ${adBoostLabel}

━━━ TEXT ANALYSIS OF THIS POST ━━━
Caption length    : ${textStats.captionChars} characters (${textStats.captionWords} words)
Content length    : ${textStats.contentChars} characters (${textStats.contentWords} words)
Hashtags detected : ${textStats.hashtagCount} → [${textStats.detectedHashtags.join(', ') || 'none'}]
Has emoji         : ${textStats.hasEmoji ? 'Yes' : 'No'}
Has CTA           : ${textStats.hasCTA ? 'Yes' : 'No'}
Has question      : ${textStats.hasQuestion ? 'Yes' : 'No'}
Has price/offer   : ${textStats.hasPricing ? 'Yes' : 'No'}

━━━ AI PREDICTIONS ━━━
Likes                : ${likesVal.toLocaleString()} (dataset avg: ${DATASET_AVERAGES.likes.toLocaleString()} → this is ${vsAvgLikes} average)
Comments             : ${commentsVal.toLocaleString()} (dataset avg: ${DATASET_AVERAGES.comments.toLocaleString()} → this is ${vsAvgComments} average)
Shares               : ${sharesVal.toLocaleString()}
Clicks               : ${clicksVal.toLocaleString()} (dataset avg: ${DATASET_AVERAGES.clicks.toLocaleString()} → this is ${vsAvgClicks} average)
Timing Quality Score : ${tqsPercent}% (dataset peak: 82% at 6–9 PM best days)

━━━ TIMING DIAGNOSIS (from dataset) ━━━
Current schedule verdict : ${timingAnalysis.verdict}
Dataset peak window      : ${BEST_DAYS[platform]?.[0]}/${BEST_DAYS[platform]?.[1]} at ${bestWindow.hours}
Best window reasoning    : ${bestWindow.reasoning}

━━━ DATASET BENCHMARKS FOR IMPROVEMENT ━━━
Hashtag impact (from 125,000 posts):
  • Current (${textStats.hashtagCount} hashtags, bucket "${textStats.hashtagBucket}"): likes≈${textStats.hashtagBenchmark.likes.toLocaleString()}, comments≈${textStats.hashtagBenchmark.comments}, clicks≈${textStats.hashtagBenchmark.clicks.toLocaleString()}
  • Optimal (7–9 hashtags): likes≈${hashtagPotential.likes.toLocaleString()}, comments≈${hashtagPotential.comments}, clicks≈${hashtagPotential.clicks.toLocaleString()}
  ${textStats.hashtagCount < 7 ? `→ Adding ${7 - textStats.hashtagCount} more relevant hashtags could multiply engagement significantly` : '→ Hashtag count is in optimal range'}

Ad Boost impact (from dataset):
  • Without boost: likes≈${AD_BOOST_IMPACT.withoutBoost.likes.toLocaleString()}, clicks≈${AD_BOOST_IMPACT.withoutBoost.clicks.toLocaleString()}, TQS≈44%
  • With boost: likes≈${AD_BOOST_IMPACT.withBoost.likes.toLocaleString()}, clicks≈${AD_BOOST_IMPACT.withBoost.clicks.toLocaleString()}, TQS≈61%
  ${isAdBoosted ? '→ This post IS boosted — maximise ROI with correct targeting' : '→ This post is NOT boosted — boosting could give 7× likes and 4.6× clicks'}

Content length (51–100 chars outperforms 0–50 by 3×):
  ${textStats.contentChars < 51 ? `→ Content is only ${textStats.contentChars} chars — expanding to 51–100 chars could triple engagement` : `→ Content length (${textStats.contentChars} chars) is in a good range`}

━━━ YOUR TASK ━━━
Generate a JSON explainability report for this SPECIFIC post. Reference the actual caption text in each tip.
Use EXACTLY these values for best_posting_time (do NOT change them):
  recommended_days  : ${JSON.stringify(bestWindow.days)}
  recommended_hours : "${bestWindow.hours}"
  reasoning         : use the timing rationale provided above, referencing the ${category} category and ${platform}

JSON schema:
{
  "overall_assessment": "2-3 sentences referencing the actual caption/content words and comparing to dataset averages",
  "performance_level": "Low | Moderate | Good | Excellent",
  "improvements": [
    {
      "metric": "Likes | Comments | Shares | Clicks | Timing Quality Score",
      "current_score": "predicted value with unit",
      "improvement_tips": [
        "tip that quotes or references the specific caption/content",
        "dataset-backed actionable tip with numbers",
        "Sri Lanka-specific tip for ${category} businesses"
      ]
    }
  ],
  "caption_advice": "Rewrite or specific edits quoting the EXACT current caption words and what to change/add",
  "hashtag_suggestions": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5", "#tag6", "#tag7", "#tag8", "#tag9"],
  "content_quality_tips": [
    "specific visual/format tip for ${category} content on ${platform}",
    "second tip tied to the actual content description"
  ],
  "best_posting_time": {
    "recommended_days": ${JSON.stringify(bestWindow.days)},
    "recommended_hours": "${bestWindow.hours}",
    "reasoning": "${bestWindow.reasoning.replace(/"/g, "'")}"
  },
  "platform_specific_tips": [
    "specific ${platform} algorithm tip for ${category}",
    "second ${platform} tip with a data-backed number"
  ],
  "ad_boost_advice": "Specific boosting advice for ${category} on ${platform} with LKR budget range and targeting parameters",
  "novelty_insight": "One unique, non-obvious insight about THIS specific post (quote the caption) that a typical Sri Lankan SME would not know"
}`;

  const response = await axios.post(
    GROQ_API,
    {
      model: 'llama-3.3-70b-versatile',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user',   content: userPrompt },
      ],
      temperature: 0.85,
      max_tokens: 2800,
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

  // ── Enforce dataset-computed best_posting_time (never let LLM override) ──
  parsed.best_posting_time = {
    recommended_days:  bestWindow.days,
    recommended_hours: bestWindow.hours,
    reasoning: parsed.best_posting_time?.reasoning || bestWindow.reasoning,
  };

  return parsed;
}

module.exports = { generateInsights };
