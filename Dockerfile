# Install pyenv dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
    python3-openssl git

# Install pyenv
RUN curl https://pyenv.run | bash

# Set pyenv environment variables
ENV PYENV_ROOT="$HOME/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"

# Install Python 3.10 using pyenv
RUN pyenv install 3.10.0 && \
    pyenv global 3.10.0

# Rehash pyenv to apply changes
RUN pyenv rehash
