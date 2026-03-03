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
  CONTENT_LENGTH_BENCHMARKS,
  TRENDING_IMPACT,
} = require('./datasetInsights');

/* ─── API endpoints ──────────────────────────────────────────────────────────*/
const GROQ_ENDPOINT   = 'https://api.groq.com/openai/v1/chat/completions';
const OPENAI_ENDPOINT = 'https://api.openai.com/v1/chat/completions';

/* ─── Provider definitions (tried in order) ─────────────────────────────────*/
const PROVIDERS = [
  {
    name:     'Groq-Primary',
    url:      GROQ_ENDPOINT,
    model:    'llama-3.3-70b-versatile',
    getKey:   () => process.env.GROQ_API_KEY_1 || process.env.GROQ_API_KEY,
    maxTokens: 3200,
  },
  {
    name:     'Groq-Fallback',
    url:      GROQ_ENDPOINT,
    model:    'llama-3.3-70b-versatile',
    getKey:   () => process.env.GROQ_API_KEY_2,
    maxTokens: 3200,
  },
  {
    name:     'OpenAI-GPT4o-mini',
    url:      OPENAI_ENDPOINT,
    model:    'gpt-4o-mini',
    getKey:   () => process.env.OPENAI_API_KEY,
    maxTokens: 3500,
  },
];

/**
 * Calls one provider; returns parsed JSON or throws on failure.
 */
async function callProvider(provider, messages, temperature) {
  const key = provider.getKey();
  if (!key) throw new Error(`No API key configured for ${provider.name}`);

  const response = await axios.post(
    provider.url,
    {
      model:           provider.model,
      messages,
      temperature,
      max_tokens:      provider.maxTokens,
      response_format: { type: 'json_object' },
    },
    {
      headers: {
        Authorization: `Bearer ${key}`,
        'Content-Type': 'application/json',
      },
      timeout: 55000,
    }
  );

  const raw = response.data?.choices?.[0]?.message?.content || '{}';
  return JSON.parse(raw);
}

/**
 * Tries each provider in order; returns the first successful result.
 */
async function callWithFallback(messages, temperature = 0.82) {
  let lastError;
  for (const provider of PROVIDERS) {
    try {
      console.log(`[explainService] Trying provider: ${provider.name}`);
      const result = await callProvider(provider, messages, temperature);
      console.log(`[explainService] Success with: ${provider.name}`);
      return result;
    } catch (err) {
      lastError = err;
      console.warn(`[explainService] ${provider.name} failed: ${err.message}`);
    }
  }
  throw new Error(`All AI providers failed. Last error: ${lastError?.message}`);
}

/* ──────────────────────────────────────────────────────────────────────────── */

/**
 * Generates deep explainability + recommendations for a Sri Lankan SME post.
 * Grounded in pre-computed dataset statistics (no raw dataset sent to LLM).
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

  const vsAvgLikes    = likesVal    > DATASET_AVERAGES.likes    ? 'above' : 'below';
  const vsAvgComments = commentsVal > DATASET_AVERAGES.comments ? 'above' : 'below';
  const vsAvgClicks   = clicksVal   > DATASET_AVERAGES.clicks   ? 'above' : 'below';

  const hashtagPotential       = HASHTAG_BENCHMARKS['7-9'];
  const contentLenOptimal      = CONTENT_LENGTH_BENCHMARKS['51-100'];
  const contentLenCurrent      = CONTENT_LENGTH_BENCHMARKS['0-50'];
  const likesGapIfOptimalHours = Math.round((DATASET_AVERAGES.likes * (0.82 / 0.513)) - likesVal);

  // ── Unique post fingerprint ───────────────────────────────────────────────
  const captionWords = (caption || '').trim().split(/\s+/).filter(Boolean);
  const fingerprint  = [
    captionWords.slice(0, 6).join(' ') || 'no-caption',
    category,
    platform,
    `${timingAnalysis.dayName}-H${timingAnalysis.hour}`,
    `tqs${tqsPercent}`,
    `L${likesVal}`,
    `ht${textStats.hashtagCount}`,
    isAdBoosted ? 'boosted' : 'organic',
    `f${followers}`,
  ].join('|');

  // ── Peak-time behavioural facts for this platform (drive LLM reasoning) ──
  const peakDays  = (BEST_DAYS[platform]  || BEST_DAYS['Facebook']).slice(0, 4);
  const peakHours = (BEST_HOURS[platform] || BEST_HOURS['Facebook']).slice(0, 4);
  const peakHourRange = `${peakHours[peakHours.length - 1]}:00 – ${peakHours[0] + 1}:00`;

  // Platform-specific audience behaviour data
  const platformBehaviourFacts = {
    Facebook:  `Sri Lankan Facebook users are most active between 6–9 PM on weekdays, especially Wednesday and Monday. After-work browsing peaks at this window as users unwind. Sunday afternoons see a secondary browsing spike as families plan the week ahead. Engagement drops by ~57% outside this window per research data.`,
    Instagram: `Sri Lankan Instagram users exhibit peak activity on Sunday and Monday evenings (7–9 PM) driven by leisure scrolling and weekly planning. Tuesday also shows strong performance due to mid-week inspiration seeking. TQS research data confirms 6–9 PM consistently yields the highest engagement rates of the day.`,
    TikTok:    `TikTok in Sri Lanka peaks sharply on Saturday and Sunday evenings (6–9 PM) when users have leisure time for entertainment content. Friday evenings see a boost as the weekend starts. Research shows late-night weekend posting outperforms weekday-morning posting by 2.3× on TikTok Sri Lanka.`,
    Twitter:   `Sri Lankan Twitter users are most engaged on Sunday and Monday evenings. News-cycle-driven spikes occur mid-week (Tuesday–Wednesday) but entertainment/brand posts perform best on weekend evenings. Peak TQS of 0.82 is observed consistently at 7–9 PM on optimal days.`,
    YouTube:   `YouTube Sri Lanka peaks on Wednesday and Monday evenings (7–9 PM) when users settle in for longer-format content viewing. Sunday afternoons are strong for tutorial and review content. Research confirms the 6–9 PM window produces 2.3× more clicks than morning slots.`,
  };
  const behaviourFact = platformBehaviourFacts[platform] || platformBehaviourFacts['Facebook'];

  // ── System prompt ─────────────────────────────────────────────────────────
  const systemPrompt = `You are a senior social media research analyst specialising in Sri Lankan SME digital marketing.
You conduct evidence-based analysis on social media engagement data collected from Sri Lankan businesses.
Your role is to explain WHY predictions are what they are, and HOW the SME owner can specifically improve their results.

CRITICAL RULES — follow every one without exception:
1. CONTENT SPECIFICITY: Quote actual words or phrases from the caption and content in EVERY insight. Generic advice is forbidden.
2. PEAK TIME REASONING: Explain WHY each recommended day and hour gets more engagement — reference audience behaviour patterns, not the research dataset directly.
3. VIRAL HASHTAGS: Suggest 9–12 hashtags that are (a) specific to the caption topic, (b) trending on ${platform} in Sri Lanka, and (c) a mix of broad-reach + niche + Sinhala hashtags. Do NOT use hashtags already in the caption.
4. RECOMMENDATIONS UNIQUENESS: Every recommendation must reference at least one specific word/phrase from this post. No copy-paste advice across posts.
5. METRIC IMPROVEMENT: For each metric that is below its benchmark, give a concrete, numbered action with expected uplift percentage.
6. SINHALA AUDIENCE: Include at least 1 Sinhala or bilingual element in hashtags and recommendations for local reach.
7. BEST POSTING TIME: Use EXACTLY the days and hours injected — explain the behavioural reason WHY those times work.
8. Respond ONLY with valid JSON. No markdown, no preamble, no explanation outside JSON.`;

  // ── User prompt ───────────────────────────────────────────────────────────
  const userPrompt = `POST FINGERPRINT: ${fingerprint}

════════════════════════════════════════
EXACT POST INPUTS
════════════════════════════════════════
Platform   : ${platform}
Category   : ${category}
Caption    : "${caption || 'Not provided'}"
Content    : "${content || 'Not provided'}"
Post Date  : ${post_date} (${timingAnalysis.dayName})
Post Time  : ${post_time} (${timingAnalysis.hour}:00)
Followers  : ${Number(followers).toLocaleString()}
Ad Boost   : ${adBoostLabel}

════════════════════════════════════════
TEXT ANALYSIS OF THIS CAPTION & CONTENT
════════════════════════════════════════
Caption    : ${textStats.captionChars} chars / ${textStats.captionWords} words
Content    : ${textStats.contentChars} chars / ${textStats.contentWords} words
Hashtags   : ${textStats.hashtagCount} found → [${textStats.detectedHashtags.join(', ') || 'NONE'}]
Has emoji  : ${textStats.hasEmoji ? 'YES' : 'NO — missing (emojis add 12–18% engagement lift)'}
Has CTA    : ${textStats.hasCTA ? 'YES' : 'NO — missing ("Shop now", "DM us", "Click link in bio")'}
Has question: ${textStats.hasQuestion ? 'YES' : 'NO — missing (questions drive 32% more comments)'}
Has pricing : ${textStats.hasPricing ? 'YES' : 'NO'}

════════════════════════════════════════
AI PREDICTED METRICS
════════════════════════════════════════
Likes                : ${likesVal.toLocaleString()} (benchmark avg: ${DATASET_AVERAGES.likes.toLocaleString()} → ${vsAvgLikes} average by ${Math.abs(likesVal - DATASET_AVERAGES.likes).toLocaleString()})
Comments             : ${commentsVal.toLocaleString()} (benchmark avg: ${DATASET_AVERAGES.comments.toLocaleString()} → ${vsAvgComments} average)
Shares               : ${sharesVal.toLocaleString()} (benchmark avg: ${DATASET_AVERAGES.shares.toLocaleString()})
Clicks               : ${clicksVal.toLocaleString()} (benchmark avg: ${DATASET_AVERAGES.clicks.toLocaleString()} → ${vsAvgClicks} average)
Timing Quality Score : ${tqsPercent}% (peak achievable: 82% on best days/hours)

════════════════════════════════════════
TIMING ANALYSIS
════════════════════════════════════════
Current schedule     : ${timingAnalysis.dayName} at ${timingAnalysis.hour}:00
Verdict              : ${timingAnalysis.verdict}
Best days for ${platform.padEnd(9)}: ${peakDays.join(', ')}
Best hours for ${platform.padEnd(8)}: ${peakHourRange}
Platform behaviour   : ${behaviourFact}

════════════════════════════════════════
QUANTITATIVE IMPROVEMENT POTENTIAL
════════════════════════════════════════
HASHTAGS — current count: ${textStats.hashtagCount} (bucket "${textStats.hashtagBucket}")
  Current bucket performance  : likes≈${textStats.hashtagBenchmark.likes.toLocaleString()}, comments≈${textStats.hashtagBenchmark.comments}, clicks≈${textStats.hashtagBenchmark.clicks.toLocaleString()}
  Optimal bucket (7–9 tags)   : likes≈${hashtagPotential.likes.toLocaleString()}, comments≈${hashtagPotential.comments}, clicks≈${hashtagPotential.clicks.toLocaleString()}
  Gap if this post had 7–9 HT : ${textStats.hashtagCount < 7 ? `+${(hashtagPotential.likes - textStats.hashtagBenchmark.likes).toLocaleString()} more likes, +${(hashtagPotential.clicks - textStats.hashtagBenchmark.clicks).toLocaleString()} more clicks` : 'Already optimal'}

AD BOOST
  Current (${isAdBoosted ? 'boosted' : 'no boost'})   : likes≈${isAdBoosted ? AD_BOOST_IMPACT.withBoost.likes.toLocaleString() : AD_BOOST_IMPACT.withoutBoost.likes.toLocaleString()}, clicks≈${isAdBoosted ? AD_BOOST_IMPACT.withBoost.clicks.toLocaleString() : AD_BOOST_IMPACT.withoutBoost.clicks.toLocaleString()}
  ${isAdBoosted ? 'Ensure audience targeting is correct to maximise the 7.1× likes uplift' : 'If boosted → +7.1× likes, +4.6× clicks, TQS rises from 44% → 61%'}

CONTENT LENGTH — current: ${textStats.contentChars} chars
  ${textStats.contentChars < 51 ? `Short content (${textStats.contentChars} chars) → Expanding to 51–100 chars could give: likes≈${contentLenOptimal.likes.toLocaleString()} vs current≈${contentLenCurrent.likes.toLocaleString()} (3× uplift), clicks: ${contentLenOptimal.clicks.toLocaleString()} vs ${contentLenCurrent.clicks.toLocaleString()}` : `Content is ${textStats.contentChars} chars — within good range`}

TIMING UPLIFT — if moved to peak window
  Current TQS: ${tqsPercent}% → Peak TQS achievable: 82% → estimated additional likes if timing optimised: ~${Math.max(0, likesGapIfOptimalHours).toLocaleString()}

════════════════════════════════════════
YOUR TASK: Generate ONLY valid JSON matching the schema below.
ALL fields are mandatory. Be specific — quote actual caption/content words.
════════════════════════════════════════

JSON SCHEMA:
{
  "overall_assessment": "3 sentences: (1) what this specific caption/content signals and its predicted performance level; (2) comparison to benchmarks with numbers; (3) biggest single improvement opportunity",

  "performance_level": "Low | Moderate | Good | Excellent",

  "predicted_metrics_analysis": {
    "likes": {
      "value": ${likesVal},
      "vs_benchmark": "${vsAvgLikes} average",
      "explanation": "Why this post gets this many likes — reference caption/content signals and timing"
    },
    "comments": {
      "value": ${commentsVal},
      "vs_benchmark": "${vsAvgComments} average",
      "explanation": "Why this comment count — reference whether caption asks a question or has CTA"
    },
    "shares": {
      "value": ${sharesVal},
      "explanation": "Why shares are at this level — reference content shareability signals"
    },
    "clicks": {
      "value": ${clicksVal},
      "vs_benchmark": "${vsAvgClicks} average",
      "explanation": "Why this click count — reference CTA presence, ad boost status, content length"
    },
    "timing_quality_score": {
      "value": "${tqsPercent}%",
      "explanation": "Explain why the TQS is ${tqsPercent}% based on the actual posting day (${timingAnalysis.dayName}) and hour (${timingAnalysis.hour}:00) vs peak window"
    }
  },

  "improvements": [
    {
      "metric": "Likes",
      "current_score": "${likesVal.toLocaleString()} likes",
      "potential_score": "estimated after improvements",
      "improvement_tips": [
        "Tip 1 quoting a word from the caption — what to add/change with expected % uplift",
        "Tip 2: hashtag or emoji gap — specific to this post content with numbers",
        "Tip 3: Sri Lanka-specific tactic for ${category} businesses (e.g. reference local events, pay-day cycles, festivals)"
      ]
    },
    {
      "metric": "Comments",
      "current_score": "${commentsVal.toLocaleString()} comments",
      "potential_score": "estimated after improvements",
      "improvement_tips": [
        "Tip 1: specific question to add based on the caption topic",
        "Tip 2: engagement hook tied to content",
        "Tip 3: community/reply strategy for ${category}"
      ]
    },
    {
      "metric": "Shares",
      "current_score": "${sharesVal.toLocaleString()} shares",
      "potential_score": "estimated after improvements",
      "improvement_tips": [
        "Tip 1: what makes this specific content more shareable",
        "Tip 2: format change (reel/carousel/story) with expected share uplift",
        "Tip 3: incentive or viral hook tied to the caption topic"
      ]
    },
    {
      "metric": "Clicks",
      "current_score": "${clicksVal.toLocaleString()} clicks",
      "potential_score": "estimated after improvements",
      "improvement_tips": [
        "Tip 1: CTA improvement referencing the specific caption",
        "Tip 2: link placement strategy for ${platform}",
        "Tip 3: ad boost or targeting tip for ${category} in Sri Lanka with LKR budget range"
      ]
    },
    {
      "metric": "Timing Quality Score",
      "current_score": "${tqsPercent}%",
      "potential_score": "Up to 82% at peak window",
      "improvement_tips": [
        "Explain WHY moving to the recommended days increases TQS — audience behaviour reason",
        "Explain WHY the recommended hours are peak — what Sri Lankan audiences are doing then",
        "Scheduling tip: how to use scheduling tools for this ${category} post on ${platform}"
      ]
    }
  ],

  "caption_analysis": {
    "score": "X/10",
    "strengths": ["what is working in the current caption — quote specific words"],
    "weaknesses": ["what is missing — emoji, CTA, question, pricing signal"],
    "rewritten_caption": "Improved version of the caption that quotes/builds on the existing words, adds missing elements, stays under 220 characters"
  },

  "content_analysis": {
    "score": "X/10",
    "current_length_verdict": "${textStats.contentChars} chars — verdict and ideal target",
    "improvement_tips": [
      "Specific visual/format improvement for this ${category} content on ${platform}",
      "Second tip directly tied to the actual content description",
      "Third tip about content structure or storytelling for ${category} SMEs"
    ]
  },

  "peak_times_analysis": {
    "recommended_days": ${JSON.stringify(bestWindow.days)},
    "recommended_hours": "${bestWindow.hours}",
    "why_these_days": "Explain the audience behaviour reason WHY ${bestWindow.days[0]} and ${bestWindow.days[1]} are peak days for ${platform} — reference what Sri Lankan users are doing on those days (weekly routine, leisure patterns, pay cycles, etc.)",
    "why_these_hours": "Explain the behaviour reason WHY ${peakHourRange} is the peak hour window — what are ${platform} users in Sri Lanka doing at this time (e.g. after-work relaxation, evening scrolling, family time)",
    "category_timing_note": "${bestWindow.reasoning.replace(/"/g, "'")}",
    "current_vs_optimal": "Compare current posting time (${timingAnalysis.dayName} ${timingAnalysis.hour}:00) to the optimal window and quantify the engagement gap"
  },

  "viral_hashtags": {
    "explanation": "Why these hashtags will increase reach for this specific post",
    "broad_reach": ["#tag1", "#tag2", "#tag3 — 3 high-volume hashtags relevant to this caption topic"],
    "niche_targeted": ["#tag4", "#tag5", "#tag6 — 3 niche hashtags specific to ${category} in Sri Lanka"],
    "local_sinhala": ["#tag7", "#tag8 — 1-2 Sinhala or bilingual hashtags for local Sri Lankan audience"],
    "platform_trending": ["#tag9", "#tag10 — 2 currently trending ${platform} hashtags for ${category}"],
    "all_hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5", "#tag6", "#tag7", "#tag8", "#tag9", "#tag10", "#tag11", "#tag12"],
    "usage_tip": "How to place these hashtags in the post for maximum ${platform} algorithm reach"
  },

  "recommendations": [
    {
      "priority": "HIGH",
      "title": "Short action title specific to this post",
      "action": "Step-by-step instruction that quotes the caption/content and gives expected metric uplift with percentage",
      "expected_impact": "Quantified improvement e.g. +45% likes, +2× clicks"
    },
    {
      "priority": "HIGH",
      "title": "Second high-priority action",
      "action": "Specific action tied to this post's gaps",
      "expected_impact": "Quantified improvement"
    },
    {
      "priority": "MEDIUM",
      "title": "Medium-priority improvement",
      "action": "Caption or content specific improvement",
      "expected_impact": "Expected uplift"
    },
    {
      "priority": "MEDIUM",
      "title": "Timing optimisation",
      "action": "Move this post to the recommended window — explain why with audience behaviour",
      "expected_impact": "TQS up to 82%, estimated engagement increase"
    },
    {
      "priority": "LOW",
      "title": "Long-term strategy tip",
      "action": "One long-term brand-building recommendation specific to ${category} on ${platform} in Sri Lanka",
      "expected_impact": "Compounding engagement growth over 30–60 days"
    }
  ],

  "ad_boost_strategy": {
    "should_boost": ${isAdBoosted ? '"Already boosted — optimise targeting"' : '"Yes — significant ROI opportunity"'},
    "recommended_budget_lkr": "Specific LKR daily/weekly budget range for ${category} SME",
    "target_audience": "Detailed targeting parameters: age, location, interests specific to ${category} in Sri Lanka",
    "expected_roi": "Projected metric improvements with boost vs without",
    "boost_timing": "When to activate boost relative to the peak posting window"
  },

  "platform_specific_tips": [
    "Specific ${platform} algorithm tip for ${category} posts — reference the actual caption",
    "Second ${platform} feature to use (Reels/Stories/Carousel/Live) with data-backed engagement number",
    "Third ${platform} growth tactic specific to Sri Lankan ${category} audience"
  ],

  "novelty_insight": "One unique, non-obvious research insight about THIS specific post (quote the caption directly) that the average Sri Lankan SME would not know — something counterintuitive or data-driven that provides real competitive advantage"
}`;

  // ── Call AI with fallback chain ─────────────────────────────────────────────
  let parsed;
  try {
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user',   content: userPrompt },
    ];
    parsed = await callWithFallback(messages, 0.82);
  } catch (err) {
    console.error('[explainService] All providers failed:', err.message);
    parsed = {
      overall_assessment: 'AI explanation service is temporarily unavailable. Please try again shortly.',
      performance_level: 'Moderate',
      improvements: [],
    };
  }

  // ── Enforce dataset-computed peak times (never let LLM override) ──────────
  if (parsed.peak_times_analysis) {
    parsed.peak_times_analysis.recommended_days  = bestWindow.days;
    parsed.peak_times_analysis.recommended_hours = bestWindow.hours;
  }

  // Keep legacy best_posting_time field for backwards compatibility
  parsed.best_posting_time = {
    recommended_days:  bestWindow.days,
    recommended_hours: bestWindow.hours,
    reasoning: parsed.peak_times_analysis?.why_these_days
      || parsed.best_posting_time?.reasoning
      || bestWindow.reasoning,
  };

  // Keep legacy hashtag_suggestions field for backwards compatibility
  if (!parsed.hashtag_suggestions && parsed.viral_hashtags?.all_hashtags) {
    parsed.hashtag_suggestions = parsed.viral_hashtags.all_hashtags.slice(0, 9);
  }

  return parsed;
}

module.exports = { generateInsights };
