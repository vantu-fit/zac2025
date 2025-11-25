FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# Update package list and install system dependencies with retry logic
RUN apt-get update 

RUN apt-get install -y --fix-missing \
    libgl1-mesa-glx \
    libglib2.0-0 \
    vim \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /code

# Copy requirements first for better Docker layer caching
COPY requirements.txt .


# Update pip and install packages individually for better reliability
RUN pip install --upgrade pip setuptools wheel

# Install lighter packages first
RUN pip install --no-cache-dir --timeout 300 --retries 3 numpy tqdm python-dotenv

# Install OpenCV (can be large)
RUN pip install --no-cache-dir --timeout 600 --retries 5 opencv-python

# Install ultralytics (also large with many dependencies)
RUN pip install --no-cache-dir --timeout 600 --retries 5 ultralytics

# Install JupyterLab last (largest package)
RUN pip install --no-cache-dir --timeout 1500 --retries 5 jupyterlab

RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p  /result
RUN mkdir /data


# Copy application files
COPY . .
COPY .docker.env .env


EXPOSE 9777




