FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
WORKDIR /app/src

#CMD ["python", "main.py", "data.domain=cmn.publication.Publication", "data.source=../data/dblp/toy.dblp.v12.json", "data.output=../output/dblp/toy.dblp.v12.json", "~data.filter"]
#CMD ["python", "main.py", "data.domain=cmn.publication.Publication", "data.source=../data/dblp/dblp.v12.json", "data.output=../output/dblp/dblp.v12.json"]

#CMD ["python", "main.py", "data.domain=cmn.movie.Movie", "data.source=../data/imdb/toy.title.basics.tsv", "data.output=../output/imdb/toy.title.basics.tsv", "~data.filter"]
#CMD ["python", "main.py", "data.domain=cmn.movie.Movie", "data.source=../data/imdb/title.basics.tsv", "data.output=../output/imdb/title.basics.tsv"]

#CMD ["python", "main.py", "data.domain=cmn.repository.Repository", "data.source=../data/gith/toy.repos.csv", "data.output=../output/gith/toy.repos.csv", "~data.filter"]
#CMD ["python", "main.py", "data.domain=cmn.repository.Repository", "data.source=../data/gith/repos.csv", "data.output=../output/gith/repos.csv"]

CMD ["python", "main.py", "data.domain=cmn.patent.Patent", "data.source=../data/uspt/toy.patent.tsv", "data.output=../output/uspt/toy.patent.tsv", "~data.filter"]
#CMD ["python", "main.py", "data.domain=cmn.patent.Patent", "data.source=../data/uspt/patent.tsv", "data.output=../output/uspt/patent.tsv"]

#========================================================
#docker build -t ghcr.io/fani-lab/opentf/toy-{###}:main .
#docker run --rm -v ./data:/app/data:ro -v ./output:/app/output ghcr.io/fani-lab/opentf/toy-{###}:main

#echo {git-key} | docker login ghcr.io -u fani-lab --password-stdin

#docker push ghcr.io/fani-lab/opentf/toy-{###}:main
#docker pull ghcr.io/fani-lab/opentf/toy-{###}:main
