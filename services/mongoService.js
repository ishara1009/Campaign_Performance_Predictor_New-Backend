const mongoose = require('mongoose');

let connected = false;

async function connectMongo() {
  if (connected) return;
  const uri = process.env.MONGODB_URI;
  if (!uri) throw new Error('MONGODB_URI is not set in environment variables');
  await mongoose.connect(uri);
  connected = true;
  console.log('MongoDB connected');
}

module.exports = { connectMongo };
