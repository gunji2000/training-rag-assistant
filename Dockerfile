FROM python:3.13.5

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0"]