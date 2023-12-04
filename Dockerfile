FROM python:3.11-slim

WORKDIR /usr/src/app

# First, copy only the requirements.txt file
COPY requirements.txt .

# Install dependencies - this layer will be cached as long as requirements.txt doesn't change
RUN pip install -r requirements.txt

# Now copy the rest of your project
COPY . .

ENV PYTHONPATH /usr/src/app

EXPOSE 5000

CMD ["python", "pipeline_steps/step51_model_endpoint.py"]
