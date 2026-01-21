FROM python:latest
WORKDIR /app
RUN apt-get update \
  && apt-get install --yes --no-install-recommends nodejs npm
RUN pip install mystmd
COPY . .
RUN myst build --html
RUN npm install -g serve
CMD ["serve", "-s",  "_build/html"]

