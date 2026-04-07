FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# ✅ install extra required package
RUN pip install openenv-core

EXPOSE 7860

# ✅ IMPORTANT: run server/app.py (NOT inference)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
