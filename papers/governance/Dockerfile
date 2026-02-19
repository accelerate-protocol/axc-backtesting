FROM python:latest
WORKDIR /app
RUN apt-get update \
  && apt-get install --yes --no-install-recommends nodejs npm git \
  && rm -rf /var/lib/apt/lists/* \
  && pip install --no-cache-dir mystmd \
  && npm install -g serve \
  && npm cache clean --force \
  && rm -rf /root/.npm \
  && adduser --disabled-password --gecos '' appuser \
  && chown -R appuser:appuser /app
COPY startup.sh .
USER appuser
EXPOSE 3000

CMD ["bash", "./startup.sh"]


