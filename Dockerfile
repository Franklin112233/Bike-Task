FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py utils.py ./
COPY .streamlit ./.streamlit
COPY artifacts ./artifacts
EXPOSE 8080
ENV PYTHONUNBUFFERED=1
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
