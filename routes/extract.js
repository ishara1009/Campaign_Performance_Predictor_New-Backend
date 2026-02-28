const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const { extractTextFromImage } = require('../controllers/extractController');

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, path.join(__dirname, '../uploads')),
  filename: (req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`),
});

const upload = multer({
  storage,
  limits: { fileSize: 10 * 1024 * 1024 }, // 10 MB
  fileFilter: (req, file, cb) => {
    const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
    allowed.includes(file.mimetype) ? cb(null, true) : cb(new Error('Only image files allowed'));
  },
});

// POST /api/extract  (multipart: field "image")
router.post('/', upload.single('image'), extractTextFromImage);

module.exports = router;
