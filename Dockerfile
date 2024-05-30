FROM python:3.10-bookworm
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
COPY requirements.txt requirements.txt
COPY Makefile Makefile
RUN pip install -r requirements.txt
COPY api api 
CMD uvicorn api.api:app --reload --port $PORT --host 0.0.0.0