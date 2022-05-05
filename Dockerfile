FROM python:3.9.11

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app
ADD . /app
RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py", "--port=5000"]