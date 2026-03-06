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

/* ─── Provider definitions (tried in order) ─────────────────────────────────
 *
 * Key distribution:
 *   KEY_1 / KEY  → same key — used for even-numbered slots
 *   KEY_2        → separate key — used for odd-numbered slots
 *
 * 8b models are excluded: the prompt exceeds their per-request size limit (413).
 * Each model occupies a separate Groq rate-limit bucket, so rotating models
 * with the same key still provides fallback headroom.
 * ─────────────────────────────────────────────────────────────────────────── */
const PROVIDERS = [
  {
    // Slot 1 — KEY_1, 70b versatile (primary workhorse)
    name:      'Groq-70b-K1',
    url:       GROQ_ENDPOINT,
    model:     'llama-3.3-70b-versatile',
    getKey:    () => process.env.GROQ_API_KEY_1 || process.env.GROQ_API_KEY,
    maxTokens: 8192,
    timeout:   90000,
  },
  {
    // Slot 2 — KEY_2, 70b versatile (separate quota)
    name:      'Groq-70b-K2',
    url:       GROQ_ENDPOINT,
    model:     'llama-3.3-70b-versatile',
    getKey:    () => process.env.GROQ_API_KEY_2,
    maxTokens: 8192,
    timeout:   90000,
  },
  {
    // Slot 3 — KEY_1, llama3-70b (older model, different rate-limit bucket)
    name:      'Groq-Llama3-70b-K1',
    url:       GROQ_ENDPOINT,
    model:     'llama3-70b-8192',
    getKey:    () => process.env.GROQ_API_KEY_1 || process.env.GROQ_API_KEY,
    maxTokens: 8192,
    timeout:   90000,
  },
  {
    // Slot 4 — KEY_2, llama3-70b
    name:      'Groq-Llama3-70b-K2',
    url:       GROQ_ENDPOINT,
    model:     'llama3-70b-8192',
    getKey:    () => process.env.GROQ_API_KEY_2,
    maxTokens: 8192,
    timeout:   90000,
  },
  {
    // Slot 5 — KEY_1, Mixtral 8×7b (32k context, different bucket entirely)
    name:      'Groq-Mixtral-K1',
    url:       GROQ_ENDPOINT,
    model:     'mixtral-8x7b-32768',
    getKey:    () => process.env.GROQ_API_KEY_1 || process.env.GROQ_API_KEY,
    maxTokens: 8192,
    timeout:   90000,
  },
  {
    // Slot 6 — KEY_2, Mixtral 8×7b
    name:      'Groq-Mixtral-K2',
    url:       GROQ_ENDPOINT,
    model:     'mixtral-8x7b-32768',
    getKey:    () => process.env.GROQ_API_KEY_2,
    maxTokens: 8192,
    timeout:   90000,
  },
  {
    // Slot 7 — OpenAI gpt-4.1-nano: cheapest ($0.10/1M in · $0.40/1M out)
    // ~$0.002 per call — use first to protect the $5 budget
    name:      'OpenAI-4.1-nano',
    url:       OPENAI_ENDPOINT,
    model:     'gpt-4.1-nano',
    getKey:    () => process.env.OPENAI_API_KEY,
    maxTokens: 6000,   // cap output to limit cost; response still fits comfortably
    timeout:   120000,
  },
  {
    // Slot 8 — OpenAI gpt-4o-mini: ($0.15/1M in · $0.60/1M out)
    // ~$0.003 per call — fallback if nano is rate-limited or unavailable
    name:      'OpenAI-4o-mini',
    url:       OPENAI_ENDPOINT,
    model:     'gpt-4o-mini',
    getKey:    () => process.env.OPENAI_API_KEY,
    maxTokens: 6000,
    timeout:   150000,
  },
  {
    // Slot 9 — OpenAI gpt-4.1-mini: ($0.40/1M in · $1.60/1M out)
    // ~$0.007 per call — only reached if both cheaper models fail
    name:      'OpenAI-4.1-mini',
    url:       OPENAI_ENDPOINT,
    model:     'gpt-4.1-mini',
    getKey:    () => process.env.OPENAI_API_KEY,
    maxTokens: 6000,
    timeout:   150000,
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
      timeout: provider.timeout || 90000,
    }
  );

  const raw = response.data?.choices?.[0]?.message?.content || '{}';
  return JSON.parse(raw);
}

/**
 * Tries each provider in order; returns the first successful result.
 *
 * Error-specific delays:
 *   413 Payload Too Large  → skip immediately (no delay — model can't handle this prompt size)
 *   429 Rate Limited       → respect Retry-After header, min 3 s, max 10 s
 *   5xx / timeout          → 1 s pause before next attempt
 *   Other                  → 500 ms pause
 */
async function callWithFallback(messages, temperature = 0.72) {
  let lastError;

  for (const provider of PROVIDERS) {
    try {
      console.log(`[explainService] Trying provider: ${provider.name}`);
      const result = await callProvider(provider, messages, temperature);
      console.log(`[explainService] Success with ${provider.name}`);
      return result;
    } catch (err) {
      lastError = err;
      const status   = err.response?.status;
      const errMsg   = err.message || '';
      console.warn(`[explainService] ${provider.name} failed (${status || 'no-status'}): ${errMsg}`);

      if (status === 413) {
        // Payload too large — this model can never handle the prompt; skip immediately
        continue;
      }

      if (status === 429) {
        // Rate-limited — honour Retry-After if present, else back off progressively
        const retryAfterSec = parseInt(err.response?.headers?.['retry-after'] || '0', 10);
        const waitMs        = retryAfterSec > 0
          ? Math.min(retryAfterSec * 1000, 10_000)
          : 3_000;   // default 3 s when header is absent
        console.log(`[explainService] Rate-limited — waiting ${waitMs}ms before next provider`);
        await new Promise((r) => setTimeout(r, waitMs));
        continue;
      }

      if (status >= 500 || errMsg.includes('timeout')) {
        await new Promise((r) => setTimeout(r, 1_000));
        continue;
      }

      // Other errors (400, 401, network, etc.) — short pause
      await new Promise((r) => setTimeout(r, 500));
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
  const systemPrompt = `You are a senior digital marketing research scientist specialising in Sri Lankan SME social media performance analysis.
You produce academic-grade, evidence-based explainability reports that explain every prediction at a deep causal level.
Your analysis must be rigorous enough to be cited in a university research paper on AI explainability in marketing.

CRITICAL RULES — follow every one without exception:
1. CONTENT SPECIFICITY: Quote actual words or phrases from the caption and content in EVERY insight. Generic advice is strictly forbidden — every sentence must reference THIS specific post.
2. CAUSAL DEPTH: For every metric prediction, explain the ROOT CAUSE (what signals in the post drive the number) and the MECHANISM (the causal chain: signal → algorithm → user behaviour → metric outcome).
3. PSYCHOLOGICAL GROUNDING: Identify every psychological trigger present or missing in the caption and content. Name the principle (e.g. Cialdini's Scarcity, Social Proof, Authority) and explain how it applies to THIS post.
4. LINGUISTIC ANALYSIS: Analyse the caption and content for tone, sentiment, language mix (Sinhala/English ratio), readability, and emotional resonance — all referenced to actual words.
5. STRUCTURAL SCORING: Score each element of the caption (hook strength, CTA clarity, emoji, urgency, question, hashtag placement) on a 0–10 scale with a reason.
6. PEAK TIME REASONING: Explain WHY each recommended day and hour gets more engagement — reference Sri Lankan audience behaviour patterns, weekly routines, and pay cycles.
7. VIRAL HASHTAGS: Suggest 9–12 hashtags — (a) specific to the caption topic, (b) trending on ${platform} in Sri Lanka, (c) mix of broad-reach + niche + Sinhala. Do NOT use hashtags already in the caption.
8. ENGAGEMENT BARRIERS: Identify specific friction points that prevent users from engaging — cognitive load, missing urgency, unclear value proposition, etc.
9. COMPETITIVE CONTEXT: Explain how this post compares to top-performing posts in the ${category} category on ${platform} in Sri Lanka.
10. PLATFORM ALGORITHM: For platform tips, explain the specific algorithmic mechanism (reach decay, engagement window, distribution boost) that makes each tip effective.
11. SINHALA AUDIENCE: Include at least 1 Sinhala or bilingual element in hashtags, examples, and recommendations for local reach.
12. NOVELTY: Provide one non-obvious, counterintuitive research insight specific to THIS post.
13. Respond ONLY with valid JSON. No markdown, no preamble, no explanation outside JSON.`;

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
ALL fields are mandatory. Be specific — quote actual caption/content words. Write like a research paper.
════════════════════════════════════════

JSON SCHEMA:
{
  "overall_assessment": "4–5 sentences: (1) what this specific caption/content signals and its predicted performance level with numbers; (2) comparison to benchmarks with exact figures; (3) the primary causal driver of this performance level — reference the caption/content directly; (4) the single biggest improvement opportunity with quantified expected gain; (5) research context for this type of post in Sri Lankan SME marketing",

  "performance_level": "Low | Moderate | Good | Excellent",

  "linguistic_analysis": {
    "tone": "One of: Informative | Promotional | Emotional | Urgent | Conversational | Inspirational",
    "tone_explanation": "Why this tone is present — quote the specific words that set the tone. Is this tone optimal for ${category} on ${platform}? What tone would perform better and why?",
    "sentiment": "Positive | Neutral | Negative",
    "sentiment_impact": "How the detected sentiment statistically influences engagement probability for ${category} posts. Reference the specific words that carry sentiment polarity.",
    "language_mix": "e.g. 80% English, 20% Sinhala — quote the Sinhala words found. What is the ideal mix for ${platform} in Sri Lanka and why?",
    "readability_verdict": "Easy / Moderate / Complex — explain why by referencing specific phrases. What is the Flesch-Kincaid grade level equivalent and does it match the target audience literacy level?",
    "bilingual_effectiveness": "Does the current Sinhala/English mix maximise reach? Explain the code-switching effect on Sri Lankan audience engagement — reference specific bilingual phrases if present.",
    "keyword_density": "What are the 3 most prominent keyword themes in the caption/content? Are they algorithmically relevant for ${platform} ${category} search/discovery? Quote the keywords."
  },

  "psychological_triggers": {
    "present": [
      {
        "trigger": "Name of psychological principle (e.g. Social Proof, Scarcity, Authority, Reciprocity, Liking, Unity)",
        "evidence": "Quote the exact word or phrase from caption/content that activates this trigger",
        "strength": "Strong | Moderate | Weak",
        "strength_reason": "Why this trigger fires at this strength — what is missing that would make it stronger?",
        "metric_impact": "Which specific metric this trigger increases and by approximately how much"
      }
    ],
    "missing": [
      {
        "trigger": "Name of missing psychological principle",
        "why_it_matters": "Explain the psychological mechanism — why does THIS trigger increase engagement for THIS type of post?",
        "how_to_add": "Exact example sentence of how to incorporate it into this caption or content — must build on the existing words",
        "expected_uplift": "Estimated metric improvement e.g. +28% comments, +15% shares"
      },
      {
        "trigger": "Second missing trigger",
        "why_it_matters": "Psychological mechanism specific to ${category} audience",
        "how_to_add": "Exact example sentence",
        "expected_uplift": "Estimated impact"
      },
      {
        "trigger": "Third missing trigger",
        "why_it_matters": "Why this matters specifically for Sri Lankan SME ${category} context",
        "how_to_add": "Concrete example in the context of this caption",
        "expected_uplift": "Estimated impact"
      }
    ]
  },

  "engagement_psychology": {
    "current_motivation": "What psychological motivation does this post currently activate? (e.g. curiosity, aspiration, FOMO, desire) — quote the triggering words",
    "friction_points": "What specific psychological barriers prevent users from engaging? List each friction point with the exact element in the caption/content causing it (e.g. 'No urgency word — reader can delay action indefinitely')",
    "desired_action": "What action does the post want the reader to take? Is the post explicitly and clearly asking for it? Quote the CTA or explain why it is absent.",
    "cognitive_load": "How much mental effort is required to understand and act on this post? Is the message clear in under 3 seconds? Reference specific complex phrases if any.",
    "emotional_hook": "What emotion does this post evoke? (excitement, trust, curiosity, desire, community) Is this the optimal emotion for driving \${metric} engagement on ${platform}? Quote the emotional trigger words."
  },

  "predicted_metrics_analysis": {
    "likes": {
      "value": ${likesVal},
      "vs_benchmark": "${vsAvgLikes} average by ${Math.abs(likesVal - DATASET_AVERAGES.likes).toLocaleString()}",
      "root_cause": "The primary signal in this post that drives the likes prediction to this level — quote the specific caption/content element responsible",
      "mechanism": "Causal chain: [caption/content signal] → [user psychological response] → [${platform} algorithm behaviour] → [likes outcome]",
      "explanation": "Full explanation referencing caption/content signals, timing, and follower count"
    },
    "comments": {
      "value": ${commentsVal},
      "vs_benchmark": "${vsAvgComments} average",
      "root_cause": "Why this comment count — is there a question, debate trigger, or community hook? Quote or explain the absence.",
      "mechanism": "Causal chain from caption structure to comment behaviour on ${platform}",
      "explanation": "Full explanation referencing question presence, CTA clarity, and community engagement signals"
    },
    "shares": {
      "value": ${sharesVal},
      "root_cause": "What makes or prevents sharing of this specific content — quote the shareability signal or identify what is missing",
      "mechanism": "Causal chain: [shareability signal] → [social identity motivation] → [share action]",
      "explanation": "Why shares are at this level — reference content value, novelty, and format"
    },
    "clicks": {
      "value": ${clicksVal},
      "vs_benchmark": "${vsAvgClicks} average",
      "root_cause": "Why this click count — is there a CTA, link reference, ad boost? Quote or explain absence.",
      "mechanism": "Causal chain from CTA/content → curiosity/intent → click action on ${platform}",
      "explanation": "Full explanation referencing CTA presence, ad boost status, content length, and link visibility"
    },
    "timing_quality_score": {
      "value": "${tqsPercent}%",
      "root_cause": "Why the TQS is ${tqsPercent}% — what is the posting day/hour contributing or failing to contribute?",
      "mechanism": "How posting time affects ${platform} algorithm distribution window and audience active-session overlap",
      "explanation": "Detailed explanation based on ${timingAnalysis.dayName} at ${timingAnalysis.hour}:00 vs peak window — quantify the audience availability gap"
    }
  },

  "improvements": [
    {
      "metric": "Likes",
      "current_score": "${likesVal.toLocaleString()} likes",
      "potential_score": "Estimated after all improvements applied",
      "root_cause": "The primary reason likes are at ${likesVal} — reference specific caption/content elements",
      "improvement_tips": [
        "Tip 1 — quote a specific word from the caption — state WHAT to change, HOW to change it with an example, WHY it works (psychological/algorithmic reason), and expected % uplift",
        "Tip 2 — hashtag or emoji gap — specific to this post content with exact numbers and expected likes gain",
        "Tip 3 — Sri Lanka-specific tactic for ${category} businesses referencing local events, pay-day cycles (25th of month), or festivals (Avurudu, Vesak, Poson)"
      ]
    },
    {
      "metric": "Comments",
      "current_score": "${commentsVal.toLocaleString()} comments",
      "potential_score": "Estimated after all improvements",
      "root_cause": "The specific structural reason this post generates/fails to generate comments — quote the caption",
      "improvement_tips": [
        "Tip 1 — specific question to add based on the caption topic with the exact question text and expected comment uplift %",
        "Tip 2 — engagement hook or debate trigger tied to the actual content with example text",
        "Tip 3 — community/reply strategy for ${category} on ${platform} in Sri Lanka with a concrete example"
      ]
    },
    {
      "metric": "Shares",
      "current_score": "${sharesVal.toLocaleString()} shares",
      "potential_score": "Estimated after all improvements",
      "root_cause": "Why this content is/is not share-worthy — reference what sharing motivation is missing",
      "improvement_tips": [
        "Tip 1 — what makes this specific content more shareable, referencing the exact content topic",
        "Tip 2 — format change recommendation (Reel/Carousel/Story) with specific expected share uplift % and why that format drives shares",
        "Tip 3 — incentive or viral hook tied directly to the caption topic with example text"
      ]
    },
    {
      "metric": "Clicks",
      "current_score": "${clicksVal.toLocaleString()} clicks",
      "potential_score": "Estimated after all improvements",
      "root_cause": "Why clicks are at this level — reference CTA presence/absence and link visibility directly",
      "improvement_tips": [
        "Tip 1 — CTA improvement referencing the specific caption wording with before/after example and expected % uplift",
        "Tip 2 — link placement strategy specific to ${platform} algorithm behaviour with step-by-step instruction",
        "Tip 3 — ad boost or targeting tip for ${category} in Sri Lanka with specific LKR daily budget range and expected clicks gain"
      ]
    },
    {
      "metric": "Timing Quality Score",
      "current_score": "${tqsPercent}%",
      "potential_score": "Up to 82% at peak window",
      "root_cause": "Why TQS is ${tqsPercent}% — what the combination of ${timingAnalysis.dayName} and ${timingAnalysis.hour}:00 means for audience availability on ${platform}",
      "improvement_tips": [
        "Explain WHY moving to the recommended days increases TQS to 82% — cite the specific audience behaviour pattern on those days (e.g. after-work routine, weekend leisure, weekly planning)",
        "Explain WHY the recommended hours are peak — what are Sri Lankan ${platform} users specifically doing during those hours, and why is that activity state optimal for ${category} content",
        "Provide a scheduling workflow: which ${platform} scheduling tool to use, when to set the post live, and why this ${category} content benefits from consistent peak-time posting over 30 days"
      ]
    }
  ],

  "caption_analysis": {
    "score": "X/10",
    "structural_scoring": {
      "hook_strength": "X/10 — are the first 3–5 words attention-grabbing? Quote them and explain why they work or fail",
      "cta_clarity": "X/10 — how clear and direct is the call-to-action? Quote it or explain its absence",
      "emoji_usage": "X/10 — are emojis used effectively? Quote or note absence and explain the 12–18% engagement lift opportunity",
      "urgency_level": "X/10 — is there time pressure or scarcity language? Quote or explain what is missing",
      "question_engagement": "X/10 — does the caption ask a question? Quote it or explain the 32% comment-boost opportunity missed",
      "hashtag_placement": "X/10 — are hashtags appropriate in count and placed correctly for ${platform}?"
    },
    "strengths": ["Quote the specific words or phrases that are working and explain WHY each one is effective"],
    "weaknesses": ["Quote or identify each specific gap — emoji: present/absent; CTA: present/absent; question: present/absent; urgency: present/absent; pricing: present/absent — with expected impact of each fix"],
    "language_effectiveness": "Analyse the Sinhala/English mix in the caption. Does code-switching enhance or reduce clarity for the target ${category} audience in Sri Lanka? Quote specific bilingual phrases and rate their effectiveness.",
    "rewritten_caption": "Improved version of the caption that quotes/builds on the existing words, adds missing elements (emoji, CTA, question, urgency), under 220 characters"
  },

  "content_analysis": {
    "score": "X/10",
    "current_length_verdict": "${textStats.contentChars} characters — is this optimal? State the ideal range for ${category} on ${platform} and why",
    "storytelling_analysis": "Does the content tell a story? What narrative structure is present or missing? Reference the actual content words and explain what narrative arc would increase shares.",
    "value_proposition": "Is the value clearly communicated to the ${category} audience? What is the reader gaining from this post? Quote the value statement or explain why it is absent.",
    "visual_content_recommendation": "What specific type of image or video should accompany this content for maximum engagement? Explain why this format works for ${category} on ${platform} with an expected engagement uplift.",
    "improvement_tips": [
      "Specific visual/format improvement for this ${category} content on ${platform} — include a concrete example and expected % uplift",
      "Second tip directly tied to the actual content description with before/after example",
      "Third tip about content structure or storytelling for ${category} SMEs in Sri Lanka — include a cultural or local context angle"
    ]
  },

  "caption_content_explainability": {
    "caption_word_analysis": {
      "original": "${(caption || 'Not provided').replace(/"/g, "'")}",
      "what_is_working": [
        {
          "element": "Quote the specific word or phrase that is effective",
          "why_it_works": "The psychological or algorithmic reason this element drives engagement — name the principle (e.g. Specificity Bias, Authority Cue, Emotional Resonance)",
          "impact": "Which metric it positively affects and by approximately how much, and why"
        },
        {
          "element": "Second effective element — quote it",
          "why_it_works": "Reason with named psychological/algorithmic principle",
          "impact": "Metric affected and estimated magnitude"
        }
      ],
      "what_is_missing": [
        {
          "missing_element": "Specific missing element (e.g. urgency word, emoji, CTA, price anchor, social proof)",
          "why_add_it": "Deep reason — explain the psychological mechanism and the ${platform} algorithm signal, tied to THIS caption topic",
          "example": "Show exactly how to add it — provide the modified caption sentence with the element inserted",
          "expected_uplift": "e.g. +22% comments, +15% clicks — explain why this magnitude"
        },
        {
          "missing_element": "Second missing element",
          "why_add_it": "Mechanism specific to the caption topic and ${category} audience",
          "example": "Concrete example sentence building on the existing caption words",
          "expected_uplift": "Estimated metric improvement with reason"
        },
        {
          "missing_element": "Third missing element",
          "why_add_it": "Reason specific to ${category} Sri Lankan audience cultural context",
          "example": "Concrete bilingual or culturally-resonant example",
          "expected_uplift": "Estimated metric improvement"
        }
      ],
      "improved_versions": [
        {
          "version_label": "Engagement-Optimised",
          "caption": "Rewritten caption adding emoji, CTA, and question — under 220 chars — must quote/build on existing words",
          "changes_made": ["Change 1: what was added/changed and the exact psychological/algorithmic reason WHY it improves engagement", "Change 2: same format", "Change 3: same format"]
        },
        {
          "version_label": "Sales-Focused",
          "caption": "Rewritten caption focused on driving clicks and conversions — under 220 chars — must quote/build on existing words",
          "changes_made": ["Change 1: what was added/changed and WHY it drives more clicks", "Change 2: same format"]
        },
        {
          "version_label": "Sinhala-Boosted",
          "caption": "Rewritten caption with additional Sinhala phrases to maximise local Sri Lankan reach — under 220 chars",
          "changes_made": ["Change 1: what Sinhala element was added and WHY it increases reach with local audience", "Change 2: same format"]
        }
      ]
    },
    "content_word_analysis": {
      "original": "${(content || 'Not provided').replace(/"/g, "'")}",
      "what_is_working": [
        {
          "element": "Quote the specific word or phrase from the content that is effective",
          "why_it_works": "Why this resonates with ${platform} ${category} audience — name the principle",
          "impact": "Which metric is positively affected"
        }
      ],
      "what_is_missing": [
        {
          "missing_element": "Specific missing element (social proof, scarcity, benefit statement, visual description, local reference)",
          "why_add_it": "Deep reason — psychological mechanism why this would improve performance for THIS specific content topic",
          "example": "Exact sentence or phrase to add, building on the existing content words",
          "expected_uplift": "Estimated engagement impact with reason"
        },
        {
          "missing_element": "Second missing element",
          "why_add_it": "Mechanism tied to the content topic and ${category} context",
          "example": "Concrete addition building on existing content",
          "expected_uplift": "Estimated impact"
        },
        {
          "missing_element": "Third missing element",
          "why_add_it": "Cultural or local Sri Lankan context reason",
          "example": "Example with bilingual or cultural element",
          "expected_uplift": "Estimated impact"
        }
      ],
      "improved_versions": [
        {
          "version_label": "Trust-Building",
          "content": "Rewritten content adding social proof, benefit statements, and scarcity — must build on the original words",
          "changes_made": ["Change 1: what was added and the psychological reason it builds trust and drives engagement", "Change 2: same format", "Change 3: same format"]
        },
        {
          "version_label": "Story-Driven",
          "content": "Rewritten content using a short narrative or customer scenario for maximum shares — build on original words",
          "changes_made": ["Change 1: what narrative element was added and WHY it improves shareability", "Change 2: same format"]
        }
      ]
    },
    "combined_caption_content_score": {
      "score": "X/10",
      "summary": "Overall verdict on how well the caption and content work together — quote specific words from both and explain the synergy or misalignment",
      "alignment_issue": "Does the caption promise something the content does not deliver, or vice versa? Be specific — quote both the caption and content to demonstrate the gap.",
      "top_3_priority_actions": [
        {
          "rank": 1,
          "action": "Most impactful single change to caption or content — be very specific with a before/after example quoting actual words",
          "why": "Exact causal reason this is the highest-priority action for THIS post — reference the metric it most impacts",
          "expected_impact": "Quantified metric uplift e.g. +38% likes — explain why this magnitude"
        },
        {
          "rank": 2,
          "action": "Second most impactful change with before/after example",
          "why": "Causal reason tied to this post's specific gaps",
          "expected_impact": "Quantified metric uplift with reason"
        },
        {
          "rank": 3,
          "action": "Third most impactful change with before/after example",
          "why": "Causal reason tied to this post",
          "expected_impact": "Quantified metric uplift with reason"
        }
      ]
    }
  },

  "peak_times_analysis": {
    "recommended_days": ${JSON.stringify(bestWindow.days)},
    "recommended_hours": "${bestWindow.hours}",
    "why_these_days": "Detailed audience behaviour explanation for WHY ${bestWindow.days[0]} and ${bestWindow.days[1]} are peak days on ${platform} in Sri Lanka — reference specific weekly routines, leisure patterns, pay cycles, and screen time habits that drive higher engagement on these days vs others",
    "why_these_hours": "Detailed explanation of WHY the ${peakHourRange} window is peak on ${platform} in Sri Lanka — what are users doing during this time (after-work relaxation, family meals, evening scroll), why that mental state makes them more receptive to ${category} content, and what happens to engagement outside this window",
    "missed_opportunity_cost": "What engagement is this post missing by being posted at ${timingAnalysis.dayName} ${timingAnalysis.hour}:00 instead of the optimal window? Quantify the estimated likes, clicks, and TQS difference.",
    "category_timing_note": "${bestWindow.reasoning.replace(/"/g, "'")}",
    "current_vs_optimal": "Side-by-side comparison: current schedule (${timingAnalysis.dayName} ${timingAnalysis.hour}:00, TQS: ${tqsPercent}%) vs optimal window — quantify the full engagement gap across all metrics"
  },

  "viral_hashtags": {
    "explanation": "Why these hashtags will increase reach for this specific post — explain the hashtag discovery mechanism on ${platform} for ${category} content",
    "broad_reach": ["#tag1 — brief reason why high-volume", "#tag2 — brief reason", "#tag3 — brief reason"],
    "niche_targeted": ["#tag4 — specific to ${category} Sri Lanka with reason", "#tag5 — same", "#tag6 — same"],
    "local_sinhala": ["#tag7 — Sinhala hashtag with transliteration and reach reason", "#tag8 — same"],
    "platform_trending": ["#tag9 — currently trending ${platform} hashtag for ${category} with reason", "#tag10 — same"],
    "all_hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5", "#tag6", "#tag7", "#tag8", "#tag9", "#tag10", "#tag11", "#tag12"],
    "usage_tip": "Step-by-step instruction on where to place these hashtags in the ${platform} post for maximum algorithm reach — include advice on caption vs comment placement for ${platform}"
  },

  "recommendations": [
    {
      "priority": "HIGH",
      "title": "Short action title referencing THIS post's specific gap",
      "action": "Step-by-step instruction quoting the caption/content with before/after example and expected metric uplift with percentage",
      "expected_impact": "Quantified improvement e.g. +48% likes, +2.3× clicks — explain the causal mechanism"
    },
    {
      "priority": "HIGH",
      "title": "Second high-priority action",
      "action": "Specific action tied to this post's gaps with step-by-step instruction and example",
      "expected_impact": "Quantified improvement with mechanism"
    },
    {
      "priority": "HIGH",
      "title": "Third high-priority action",
      "action": "Timing or hashtag optimisation with specific implementation steps",
      "expected_impact": "Quantified improvement"
    },
    {
      "priority": "MEDIUM",
      "title": "Medium-priority caption or content improvement",
      "action": "Specific improvement with example tied to the post topic",
      "expected_impact": "Expected uplift"
    },
    {
      "priority": "MEDIUM",
      "title": "Format or visual upgrade",
      "action": "Recommendation on post format change (Reel/Carousel/Story) specific to this ${category} content on ${platform}",
      "expected_impact": "Expected engagement improvement"
    },
    {
      "priority": "LOW",
      "title": "Long-term brand strategy",
      "action": "One long-term brand-building recommendation for ${category} on ${platform} in Sri Lanka — reference a compounding strategy tied to the post topic",
      "expected_impact": "Compounding engagement growth over 30–60 days with estimated trajectory"
    }
  ],

  "ad_boost_strategy": {
    "should_boost": ${isAdBoosted ? '"Already boosted — optimise targeting"' : '"Yes — high ROI opportunity for this post"'},
    "recommended_budget_lkr": "Specific LKR daily and weekly budget range for a ${category} SME — explain why this range maximises ROI vs overspend",
    "target_audience": "Detailed targeting parameters: age range, gender split, location (city/region), interests, behaviours — all specific to ${category} in Sri Lanka and grounded in this post's content",
    "expected_roi": "Projected metric improvements with boost vs without — use the 7.1× likes / 4.6× clicks multiplier and show the calculation",
    "boost_timing": "Exactly when to activate the boost relative to the peak posting window and why — include how many hours before peak to launch"
  },

  "platform_specific_tips": [
    {
      "tip": "Short tip title",
      "detail": "Full explanation of what to do — reference the actual caption and content",
      "algorithm_reason": "The specific ${platform} algorithm mechanism that makes this tip effective — e.g. reach decay window, engagement velocity, content distribution boost",
      "implementation": "Step-by-step how to implement this on ${platform}",
      "expected_impact": "Metric improvement estimate"
    },
    {
      "tip": "Second tip title",
      "detail": "Full explanation referencing actual post elements",
      "algorithm_reason": "Specific algorithm mechanism",
      "implementation": "Step-by-step implementation",
      "expected_impact": "Metric improvement estimate"
    },
    {
      "tip": "Third tip title",
      "detail": "Sri Lankan ${category} audience-specific tactic",
      "algorithm_reason": "Algorithm or cultural mechanism",
      "implementation": "Step-by-step implementation",
      "expected_impact": "Metric improvement estimate"
    }
  ],

  "novelty_insight": {
    "insight": "One unique, counterintuitive research finding about THIS specific post — quote the caption directly",
    "research_basis": "What data pattern or behavioural research supports this insight — be specific about what makes it non-obvious",
    "application": "Exactly how to apply this insight to THIS post — give a concrete example with the caption/content words",
    "competitive_advantage": "How applying this insight gives this ${category} SME an edge over competitors who do not know it — quantify if possible"
  }
}`;

  // ── Call AI with fallback chain ─────────────────────────────────────────────
  // Errors propagate to the controller so nothing gets saved on failure
  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user',   content: userPrompt },
  ];
  const parsed = await callWithFallback(messages, 0.72);

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

  // Normalise platform_specific_tips — new schema returns objects; keep both shapes
  if (Array.isArray(parsed.platform_specific_tips) && parsed.platform_specific_tips.length > 0) {
    if (typeof parsed.platform_specific_tips[0] === 'string') {
      // Old string format — wrap into objects for the new frontend
      parsed.platform_specific_tips = parsed.platform_specific_tips.map((tip, i) => ({
        tip: `Tip ${i + 1}`,
        detail: tip,
        algorithm_reason: '',
        implementation: '',
        expected_impact: '',
      }));
    }
  }

  // Normalise novelty_insight — new schema returns an object; keep string fallback
  if (typeof parsed.novelty_insight === 'string') {
    parsed.novelty_insight = {
      insight: parsed.novelty_insight,
      research_basis: '',
      application: '',
      competitive_advantage: '',
    };
  }

  return parsed;
}

module.exports = { generateInsights };
