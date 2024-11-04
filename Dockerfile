# Start from a Python 3.10 image
FROM python:3.10

# Set environment variables for pyenv if needed
ENV PYENV_ROOT="/root/.pyenv" \
    PATH="/root/.pyenv/bin:/root/.pyenv/shims:/root/.pyenv/versions/3.10.0/bin:$PATH"

# Copy all project files into the container
COPY . /workspace
WORKDIR /workspace

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
