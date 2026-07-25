# Multi-stage build for CrowdDNA Production
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime

# Security: Non-root user
RUN useradd -m -s /bin/bash crowddna

WORKDIR /app
COPY --from=builder /root/.local /home/crowddna/.local
COPY . /app

ENV PATH=/home/crowddna/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

RUN chown -R crowddna:crowddna /app
USER crowddna

CMD ["python", "-m", "training.run_api"]