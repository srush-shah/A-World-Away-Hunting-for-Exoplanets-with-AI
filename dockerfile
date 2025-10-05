FROM python:3.11-slim
WORKDIR /app
COPY api/deploy_api.py api/deploy_api.py
COPY model_artifacts/ model_artifacts/
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn","api.deploy_api:app","--host","0.0.0.0","--port","8000"]