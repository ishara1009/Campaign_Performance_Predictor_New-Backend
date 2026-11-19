# Campaign Performance Predictor — Backend

Node.js/Express backend for the AI-powered social media campaign performance predictor. It serves REST API endpoints for engagement prediction, AI explanations, SHAP/LIME explainability, image text extraction, and prediction history. Python inference scripts power the deep learning models (Transformer, BiGRU, BiLSTM).

---

## Table of Contents

- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [Docker](#docker)
- [API Reference](#api-reference)
  - [Health Check](#health-check)
  - [Predict](#predict)
  - [Explain](#explain)
  - [SHAP / LIME](#shap--lime)
  - [Extract Text from Image](#extract-text-from-image)
  - [History](#history)
- [Data Model](#data-model)
- [Python Scripts](#python-scripts)
- [Saved Models](#saved-models)

---

## Tech Stack

| Layer        | Technology                          |
|--------------|-------------------------------------|
| Runtime      | Node.js                             |
| Framework    | Express 4                           |
| Database     | MongoDB (Mongoose 9)                |
| Auth / BaaS  | Supabase (`@supabase/supabase-js`)  |
| File Upload  | Multer                              |
| ML Inference | Python 3 · TensorFlow CPU 2.16      |
| Explainability | SHAP ≥ 0.45 · LIME ≥ 0.2          |
| Containerisation | Docker · Docker Compose         |

---

## Project Structure

```
backend/
├── app.js                    # Express entry point
├── package.json
├── requirements.txt          # Python dependencies
├── docker-compose.yml
├── Dockerfile
│
├── controllers/
│   ├── predictController.js  # Spawns predict.py, saves to MongoDB
│   ├── explainController.js  # Generates AI insights via explainService
│   ├── extractController.js  # OCR / text extraction from uploaded images
│   ├── historyController.js  # CRUD on prediction history
│   └── shapLimeController.js # Spawns explain_shap_lime.py
│
├── routes/
│   ├── predict.js
│   ├── explain.js
│   ├── extract.js
│   ├── history.js
│   └── shapLime.js
│
├── models/
│   └── Prediction.js         # Mongoose schema
│
├── services/
│   ├── mongoService.js       # MongoDB connection
│   ├── supabaseService.js    # Supabase client helpers
│   ├── explainService.js     # AI insight generation
│   ├── textExtractorService.js
│   └── datasetInsights.js
│
├── python/
│   ├── predict.py            # Inference with Transformer.keras
│   └── explain_shap_lime.py  # SHAP KernelExplainer + LIME TabularExplainer
│
├── SavedModels/
│   ├── Transformer.keras
│   ├── Transformer_fixed.keras
│   ├── BiGRU.keras
│   ├── BiLSTM.keras
│   └── tokenizer.json
│
└── uploads/                  # Temporary image uploads (auto-created)
```

---

## Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Server
PORT=5000

# MongoDB
MONGO_URI=mongodb://localhost:27017/campaign_predictor

# Supabase
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_SERVICE_KEY=<your-service-role-key>

# Python interpreter (default: python)
PYTHON_PATH=python
```

---

## Installation

### Prerequisites

- Node.js ≥ 18
- Python ≥ 3.10
- MongoDB (local or Atlas)

### Steps

```bash
# 1. Install Node dependencies
cd backend
npm install

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Create .env (see Environment Variables above)
```

---

## Running the Server

```bash
# Development (auto-reload with nodemon)
npm run dev

# Production
npm start
```

Server starts on `http://localhost:5000` by default.

---

## Docker

```bash
# Build and start the container
docker-compose up --build

# Run in background
docker-compose up -d
```

The `docker-compose.yml` exposes port `5000`, mounts a named volume for `uploads/`, and runs a health check against `/api/health` every 30 seconds.

---

## API Reference

All endpoints are prefixed with `/api`.

### Health Check

| Method | Path         |
|--------|-------------|
| GET    | `/api/health` |

**Response**
```json
{ "status": "ok", "timestamp": "2026-05-08T10:00:00.000Z" }
```

---

### Predict

| Method | Path           |
|--------|----------------|
| POST   | `/api/predict` |

Runs the Transformer model to predict engagement metrics and saves the result to MongoDB.

**Request Body**
```json
{
  "caption":   "Your post caption",
  "content":   "Extended post content",
  "platform":  "Instagram",
  "post_date": "2026-05-10",
  "post_time": "14:00",
  "followers": 5000,
  "ad_boost":  100
}
```

**Response**
```json
{
  "success": true,
  "id": "<mongodb-document-id>",
  "prediction": {
    "likes":                142,
    "comments":             18,
    "shares":               9,
    "clicks":               74,
    "timing_quality_score": 0.83
  }
}
```

---

### Explain

| Method | Path           |
|--------|----------------|
| POST   | `/api/explain` |

Generates a plain-English AI explanation for a prediction result and updates the stored MongoDB document.

**Request Body**
```json
{
  "predictionId": "<mongodb-document-id>",
  "platform":  "Instagram",
  "post_date": "2026-05-10",
  "post_time": "14:00",
  "followers": 5000,
  "ad_boost":  100,
  "likes":     142,
  "comments":  18,
  "shares":    9,
  "clicks":    74,
  "timing_quality_score": 0.83
}
```

**Response**
```json
{
  "success": true,
  "explanation": { ... }
}
```

---

### SHAP / LIME

| Method | Path             |
|--------|-----------------|
| POST   | `/api/shap-lime` |

Runs SHAP `KernelExplainer` and LIME `TabularExplainer` + `TextExplainer` for full feature-level attributions across all five prediction targets. **Expected response time: 30–90 seconds.**

**Request Body** — same fields as `/api/predict`

**Response**
```json
{
  "success": true,
  "shapLime": {
    "shap": { ... },
    "lime": { ... },
    "summaries": [ ... ],
    "concordance": { ... }
  }
}
```

---

### Extract Text from Image

| Method | Path           | Content-Type           |
|--------|----------------|------------------------|
| POST   | `/api/extract` | `multipart/form-data`  |

Upload an image (field name: `image`) and receive the extracted text.

**Response**
```json
{
  "success": true,
  "text": "Extracted text content..."
}
```

---

### History

| Method | Path                  | Description                        |
|--------|-----------------------|------------------------------------|
| GET    | `/api/history`        | Latest 50 predictions (newest first) |
| GET    | `/api/history/:id`    | Single prediction by MongoDB ID    |
| DELETE | `/api/history/:id`    | Delete a prediction                |

**GET `/api/history` Response**
```json
{
  "success": true,
  "data": [ { ...predictionDocument }, ... ]
}
```

---

## Data Model

**`Prediction`** (MongoDB / Mongoose)

| Field                  | Type    | Notes                              |
|------------------------|---------|------------------------------------|
| `caption`              | String  | Post caption text                  |
| `content`              | String  | Extended post body                 |
| `platform`             | String  | e.g. `Instagram`, `Twitter`        |
| `post_date`            | String  | ISO date string                    |
| `post_time`            | String  | `HH:MM`                            |
| `followers`            | Number  | Account follower count             |
| `ad_boost`             | Number  | Ad spend / boost amount            |
| `likes`                | Number  | Predicted likes                    |
| `comments`             | Number  | Predicted comments                 |
| `shares`               | Number  | Predicted shares                   |
| `clicks`               | Number  | Predicted clicks                   |
| `timing_quality_score` | Number  | Score between 0 and 1              |
| `explanation`          | Mixed   | Populated after `/api/explain`     |
| `createdAt`            | Date    | Auto-managed by Mongoose           |
| `updatedAt`            | Date    | Auto-managed by Mongoose           |

---

## Python Scripts

### `python/predict.py`

Called by `predictController.js` via `child_process.spawn`.

```
Usage: python predict.py <model_path> '<json_payload>'
```

Reads the Transformer model, tokenizes text features, encodes categorical and numeric inputs, and prints a JSON object with the five predicted metrics to `stdout`.

### `python/explain_shap_lime.py`

Called by `shapLimeController.js` via `child_process.spawn`. Timeout: **180 seconds**.

```
Usage: python explain_shap_lime.py <model_path> '<json_payload>'
```

Computes:
- **SHAP** `KernelExplainer` values for all numeric features across all 5 targets
- **LIME** `TabularExplainer` importances for numeric features across all 5 targets
- **LIME** `TextExplainer` word-level attributions for caption + content
- Plain-English summaries and SHAP/LIME concordance check

---

## Saved Models

| File                      | Description                                    |
|---------------------------|------------------------------------------------|
| `Transformer.keras`       | Primary model used for inference               |
| `Transformer_fixed.keras` | Fixed/patched variant of the Transformer       |
| `BiGRU.keras`             | Bidirectional GRU model                        |
| `BiLSTM.keras`            | Bidirectional LSTM model                       |
| `tokenizer.json`          | Tokenizer vocabulary used by all text models   |
