# Image de la CLI : Python, dépendances aux versions exactes, dictionnaire NLTK et encodeurs déjà téléchargés.
# Le dataset, les embeddings et le modèle restent sur la machine hôte et sont montés dans /app/data et /app/models.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/huggingface \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

# torch en version CPU (via l'index PyTorch du fichier), sans les ~2 Go de CUDA
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

# Dictionnaire de lemmatisation et encodeurs : téléchargés à la construction, plus jamais ensuite
RUN python -m nltk.downloader -d "$NLTK_DATA" wordnet
RUN python -c "from sentence_transformers import SentenceTransformer as S; \
S('sentence-transformers/all-mpnet-base-v2'); S('microsoft/unixcoder-base')"
ENV HF_HUB_OFFLINE=1

COPY src ./src
COPY configs ./configs

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]
