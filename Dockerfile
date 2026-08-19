FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true \
    APP_RUNTIME_MODE=demo \
    AWS_REGION=us-west-2

WORKDIR /opt/data-engineering-assistant

COPY requirements.txt ./requirements.txt
COPY constraints.txt ./constraints.txt
RUN python -m pip install --no-cache-dir \
        --constraint constraints.txt \
        --requirement requirements.txt

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app \
        --create-home --home-dir /home/app app

COPY --chown=app:app knowledge/ ./knowledge/
COPY --chown=app:app config/ ./config/
COPY --chown=app:app bookkeeping/ ./bookkeeping/
COPY --chown=app:app ui/ ./ui/
COPY --chown=app:app evaluation/fixtures/ ./evaluation/fixtures/
COPY --chown=app:app evaluation/results/ ./evaluation/results/

USER 10001:10001

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).read()"]

CMD ["python", "-m", "streamlit", "run", "ui/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
