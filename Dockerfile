FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1 FINSIGHT_MODE=offline

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY web/ ./web/

EXPOSE 8000
# Run the eval gate at build/test time with:  python -m finsight.eval.run_eval --gate 0.8
CMD ["uvicorn", "finsight.api:app", "--host", "0.0.0.0", "--port", "8000"]
