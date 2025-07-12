FROM python:3.10-slim

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 5006

CMD ["panel", "serve", "server.py", "--address", "0.0.0.0", "--port", "5006", "--allow-websocket-origin=*"]