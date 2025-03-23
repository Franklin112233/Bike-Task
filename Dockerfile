FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py utils.py ./
COPY .streamlit ./.streamlit
COPY artifacts ./artifacts
EXPOSE 8501
ENV PYTHONUNBUFFERED=1
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Build: docker build -t bikehub-app .
# Run: docker run -p 8501:8501 bikehub-app
