require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

// Ensure uploads folder exists
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });

const predictRouter = require('./routes/predict');
const extractRouter = require('./routes/extract');
const explainRouter = require('./routes/explain');
const historyRouter = require('./routes/history');

const app = express();

app.use(cors({ origin: '*' }));
app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ extended: true, limit: '20mb' }));

// Static uploads
app.use('/uploads', express.static(uploadsDir));

// Routes
app.use('/api/predict', predictRouter);
app.use('/api/extract', extractRouter);
app.use('/api/explain', explainRouter);
app.use('/api/history', historyRouter);

// Health check
app.get('/api/health', (req, res) => res.json({ status: 'ok', timestamp: new Date().toISOString() }));

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Backend running on http://localhost:${PORT}`);
});