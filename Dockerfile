# ─── Base: Python 3.10 slim ───────────────────────────────────────────────────
FROM python:3.10-slim

# Install Node.js 20
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Python dependencies (heavy layer – cached unless requirements.txt changes) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Node dependencies ────────────────────────────────────────────────────────
COPY package*.json ./
RUN npm install --omit=dev

# ─── Application source ───────────────────────────────────────────────────────
COPY . .

# Ensure uploads directory exists
RUN mkdir -p /app/uploads

EXPOSE 5000

CMD ["node", "app.js"]
