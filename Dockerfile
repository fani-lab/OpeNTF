FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
WORKDIR /app/src

# Generate teamsvecs, indexes, ...
#CMD ["python", "main.py", "data.domain=cmn.publication.Publication", "data.source=../data/dblp/toy.dblp.v12.json", "data.output=../output/dblp/toy.dblp.v12.json", "~data.filter"]
#CMD ["python", "main.py", "data.domain=cmn.publication.Publication", "data.source=../data/dblp/dblp.v12.json", "data.output=../output/dblp/dblp.v12.json", "data.filter.min_nteam=10", "data.filter.min_team_size=2"]
# Generate d2v embeddings
#CMD ["python", "main.py", "data.domain=cmn.publication.Publication", "data.source=../data/dblp/dblp.v12.json", "data.output=../output/dblp/dblp.v12.json", "~data.filter", "data.embedding.class_method=mdl.emb.d2v.D2v_d2v"]
#CMD ["python", "main.py", "data.domain=cmn.publication.Publication", "data.source=../data/dblp/dblp.v12.json", "data.output=../output/dblp/dblp.v12.json", "data.filter.min_nteam=10", "data.filter.min_team_size=2", "data.embedding.class_method=mdl.emb.d2v.D2v_d2v"]

# Generate teamsvecs, indexes, ...
#CMD ["python", "main.py", "data.domain=cmn.movie.Movie", "data.source=../data/imdb/toy.title.basics.tsv", "data.output=../output/imdb/toy.title.basics.tsv", "~data.filter"]
#CMD ["python", "main.py", "data.domain=cmn.movie.Movie", "data.source=../data/imdb/title.basics.tsv", "data.output=../output/imdb/title.basics.tsv"]
# Generate d2v embeddings
#CMD ["python", "main.py", "data.domain=cmn.movie.Movie", "data.source=../data/imdb/title.basics.tsv", "data.output=../output/imdb/title.basics.tsv", "~data.filter", "data.embedding.class_method=mdl.emb.d2v.D2v_d2v"]
CMD ["python", "main.py", "data.domain=cmn.movie.Movie", "data.source=../data/imdb/title.basics.tsv", "data.output=../output/imdb/title.basics.tsv", "data.filter.min_nteam=10", "data.filter.min_team_size=2", "data.embedding.class_method=mdl.emb.d2v.D2v_d2v"]

# Generate teamsvecs, indexes, ...
#CMD ["python", "main.py", "data.domain=cmn.repository.Repository", "data.source=../data/gith/toy.repos.csv", "data.output=../output/gith/toy.repos.csv", "~data.filter"]
#CMD ["python", "main.py", "data.domain=cmn.repository.Repository", "data.source=../data/gith/repos.csv", "data.output=../output/gith/repos.csv"]
# Generate d2v embeddings

# Generate teamsvecs, indexes, ...
#CMD ["python", "main.py", "data.domain=cmn.patent.Patent", "data.source=../data/uspt/toy.patent.tsv", "data.output=../output/uspt/toy.patent.tsv", "~data.filter"]
#CMD ["python", "main.py", "data.domain=cmn.patent.Patent", "data.source=../data/uspt/patent.tsv", "data.output=../output/uspt/patent.tsv"]
# Generate d2v embeddings

#========================================================
#docker build -t ghcr.io/fani-lab/opentf/toy-{###}:main .
#docker run --rm -v ./data:/app/data:ro -v ./output:/app/output ghcr.io/fani-lab/opentf/toy-{###}:main > {###}.mt10.ts2.d2v.log 2>&1

#echo {git-key} | docker login ghcr.io -u fani-lab --password-stdin

#docker push ghcr.io/fani-lab/opentf/toy-{###}:main
#docker pull ghcr.io/fani-lab/opentf/toy-{###}:main
