FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

WORKDIR /app/src

#CMD ["python", "main.py", "data.domain=cmn.publication.Publication", "data.source=../data/dblp/toy.dblp.v12.json", "data.output=../output/dblp/toy.dblp.v12.json", "~data.filter"]
CMD ["python", "main.py", "data.domain=cmn.publication.Publication", "data.source=../data/dblp/dblp.v12.json", "data.output=../output/dblp/dblp.v12.json", "~data.filter"]

#docker build -t toy.dblp.v12.json .
#docker run --rm -v ./data:/app/data:ro -v ./output:/app/output toy.dblp.v12.json
#echo {git-key} | docker login ghcr.io -u fani-lab --password-stdin
#docker tag toy.dblp.v12.json ghcr.io/fani-lab/opentf/toy-dblp:main
#docker push ghcr.io/fani-lab/opentf/toy-dblp:main
#
#docker pull ghcr.io/fani-lab/opentf/toy-dblp:main


#CMD ["python", "main.py", "data.domain=cmn.movie.Movie", "data.source=../data/imdb/toy.title.basics.tsv", "data.output=../output/imdb/toy.title.basics.tsv", "~data.filter"]
#CMD ["python", "main.py", "data.domain=cmn.repository.Repository", "data.source=../data/gith/toy.data.csv", "data.output=../output/gith/toy.data.csv", "~data.filter"]
#CMD ["python", "main.py", "data.domain=cmn.patent.Patent", "data.source=../data/uspt/toy.patent.tsv", "data.output=../output/uspt/toy.patent.tsv", "~data.filter"]
