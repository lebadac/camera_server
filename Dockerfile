FROM python:3.10

WORKDIR /app

# Copy code & requirements
COPY distilled_student_model_weights.weights.h5 /app/distilled_student_model_weights.weights.h5
COPY main.py /app/main.py
COPY requirements.txt /app/requirements.txt
COPY start.sh /app/start.sh
COPY model/segment.py /app/model/segment.py

# Install necessary system libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install official ngrok CLI
RUN curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && \
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list && \
    apt-get update && apt-get install -y ngrok

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Give execute permission to start.sh
RUN chmod +x /app/start.sh

EXPOSE 8888

# Run the startup script
CMD ["/app/start.sh"]
