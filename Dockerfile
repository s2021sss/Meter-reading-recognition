FROM python:3.11-slim-buster

RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


COPY . .
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN python -c "from ultralytics import YOLO; YOLO('')"

RUN mkdir -p models data

RUN curl -L -o models/best.pt "https://huggingface.co/s2021sss/Meter-reading-recognition/resolve/main/best.pt"
RUN curl -L -o models/CustomCNNForDigits.pth "https://huggingface.co/s2021sss/Meter-reading-recognition/resolve/main/CustomCNNForDigits.pth"
RUN curl -L -o models/CustomCNNForUpsideDownImages.pth "https://huggingface.co/s2021sss/Meter-reading-recognition/resolve/main/CustomCNNForUpsideDownImages.pth"


ENTRYPOINT ["python", "meter-recognizer.py"]