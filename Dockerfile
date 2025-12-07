FROM vllm/vllm-openai:v0.9.1

WORKDIR /app

COPY ./app.py /app/app.py
COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt
# RUN apt-get update && apt-get install -y less && apt-get install -y zip

CMD ["python", "app.py"]