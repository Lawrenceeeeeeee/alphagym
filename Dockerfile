FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MLQUANT_DATA_ROOT=/data/lake

WORKDIR /app
COPY pyproject.toml README.md ./
RUN python -c "import pathlib,subprocess,sys,tomllib; deps=tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['dependencies']; subprocess.check_call([sys.executable,'-m','pip','install','--no-cache-dir',*deps])"
COPY src ./src
COPY scripts ./scripts
COPY config ./config
RUN python -m pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 mlquant \
    && mkdir -p /data/lake \
    && chown -R mlquant:mlquant /app /data/lake

USER mlquant
CMD ["mlquant", "--help"]
