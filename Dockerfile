FROM python:3.12-slim

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["python", "src/app/main.py"]
