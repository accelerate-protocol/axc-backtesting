FROM python:latest
WORKDIR /app
RUN apt-get update \
  && apt-get install --yes --no-install-recommends nodejs npm \
  && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mystmd
COPY . .
RUN myst build --html \
  && npm install -g serve
RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser:appuser /app
USER appuser
EXPOSE 3000

CMD ["serve", "--cors",  "_build/html/"]
#CMD ["myst", "start"]

