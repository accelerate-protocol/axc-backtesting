FROM python:latest
WORKDIR /app
RUN apt-get update \
  && apt-get install --yes --no-install-recommends nodejs npm git \
  && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mystmd
RUN git clone http://github.com/accelerate-protocol/axc-backtesting.git
RUN npm install -g serve
RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser:appuser /app
USER appuser
EXPOSE 3000

CMD ["bash", "./axc-backtesting/papers/governance/startup.sh"]


