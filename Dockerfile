# Use Python 3.12.4-slim as base image
FROM python:3.12.4-slim
# Install system dependencies, since Docker image is based on Linuix, and this is required for OpenCV and PyTorch
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install\
    libgl1\
    libgl1-mesa-glx \ 
    libglib2.0-0 -y && \
    rm -rf /var/lib/apt/lists/*
# Set the working directory to /app in container
WORKDIR /app
# Upgrade pip
RUN pip install --upgrade pip
# Copy requirements.txt file into the container at /app
COPY ./requirements.txt /app/requirements.txt
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Copy app directory contents into the container at /app
COPY ./app /app
# Make port 80 available to the world outside this container
EXPOSE 500
# Run Flask app when the container launches
CMD ["flask", "run", "--host=0.0.0.0"]