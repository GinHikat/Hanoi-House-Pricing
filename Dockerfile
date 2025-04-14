FROM python:3.12-alpine

# Install build dependencies
RUN apk add --no-cache \
    build-base \
    gfortran \
    linux-headers \
    lapack-dev \
    musl-dev \
    libffi-dev \
    openblas-dev \
    && pip install --upgrade pip

COPY . /app
WORKDIR /app
RUN pip install -r requirement.txt
CMD ["python", "app.py"]
