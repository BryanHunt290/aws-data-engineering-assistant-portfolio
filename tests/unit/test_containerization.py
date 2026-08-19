from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).parents[2]
DOCKERFILE = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
COMPOSE = PROJECT_ROOT / "compose.yaml"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_required_container_files_exist():
    assert DOCKERFILE.is_file()
    assert DOCKERIGNORE.is_file()
    assert COMPOSE.is_file()
    assert (PROJECT_ROOT / ".env.example").is_file()
    assert (
        PROJECT_ROOT / "docs" / "CONTAINERIZATION.md"
    ).is_file()


def test_dockerfile_uses_supported_official_slim_python():
    content = _read(DOCKERFILE)

    assert content.startswith("FROM python:3.12.10-slim\n")
    assert "WORKDIR /opt/data-engineering-assistant" in content
    assert "PYTHONUNBUFFERED=1" in content
    assert "PYTHONDONTWRITEBYTECODE=1" in content


def test_dockerfile_installs_dependencies_and_copies_only_runtime_code():
    content = _read(DOCKERFILE)

    assert "COPY requirements.txt ./requirements.txt" in content
    assert "COPY constraints.txt ./constraints.txt" in content
    assert "python -m pip install --no-cache-dir" in content
    assert "--constraint constraints.txt" in content
    assert "--requirement requirements.txt" in content
    assert "COPY --chown=app:app knowledge/ ./knowledge/" in content
    assert "COPY --chown=app:app config/ ./config/" in content
    assert "COPY --chown=app:app bookkeeping/ ./bookkeeping/" in content
    assert "COPY --chown=app:app ui/ ./ui/" in content
    assert (
        "COPY --chown=app:app evaluation/fixtures/ "
        "./evaluation/fixtures/"
    ) in content
    assert (
        "COPY --chown=app:app evaluation/results/ "
        "./evaluation/results/"
    ) in content
    assert "COPY . " not in content
    assert "COPY tests" not in content
    assert "ui/demo_corpus" not in _read(DOCKERIGNORE)


def test_dockerfile_runs_as_non_root_and_exposes_streamlit():
    content = _read(DOCKERFILE)

    assert "useradd --system --uid 10001" in content
    assert "USER 10001:10001" in content
    assert "USER root" not in content
    assert "EXPOSE 8501" in content
    assert "APP_RUNTIME_MODE=demo" in content
    assert "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false" in content


def test_dockerfile_command_and_health_check_are_local_only():
    content = _read(DOCKERFILE)

    assert 'CMD ["python", "-m", "streamlit", "run", "ui/app.py"' in content
    assert "--server.address=0.0.0.0" in content
    assert "--server.port=8501" in content
    assert "HEALTHCHECK" in content
    assert "http://127.0.0.1:8501/_stcore/health" in content
    health_line = next(
        line for line in content.splitlines() if "urllib.request.urlopen" in line
    )
    assert "bedrock" not in health_line.casefold()
    assert "embed" not in health_line.casefold()


def test_dockerignore_excludes_sensitive_and_generated_files():
    patterns = {
        line.strip()
        for line in _read(DOCKERIGNORE).splitlines()
        if line.strip() and not line.startswith("#")
    }

    for required in {
        ".git",
        ".venv",
        "venv",
        "**/__pycache__",
        ".pytest_cache",
        "cdk.out",
        ".env",
        ".local",
        ".aws",
        "**/.aws",
        "credentials",
        "tests",
        "out.json",
        "logs",
        "*.log",
        "tmp",
        ".idea",
        ".vscode",
    }:
        assert required in patterns


def test_compose_defaults_to_offline_demo_with_health_and_restart():
    content = _read(COMPOSE)

    assert re.search(r"(?m)^  demo:\s*$", content)
    assert "APP_RUNTIME_MODE: demo" in content
    assert '"${APP_PORT:-8501}:8501"' in content
    assert "restart: unless-stopped" in content
    assert "healthcheck:" in content
    assert "http://127.0.0.1:8501/_stcore/health" in content


def test_compose_bedrock_mode_is_explicit_profile_without_credentials():
    content = _read(COMPOSE)

    assert re.search(r"(?m)^  bedrock:\s*$", content)
    assert re.search(
        r"profiles:\s*\n\s+- bedrock",
        content,
    )
    assert "APP_RUNTIME_MODE: bedrock" in content
    assert "AWS_ACCESS_KEY_ID" not in content
    assert "AWS_SECRET_ACCESS_KEY" not in content
    assert "AWS_SESSION_TOKEN" not in content


def test_compose_qdrant_is_loopback_persistent_and_health_checked():
    content = _read(COMPOSE)

    assert re.search(r"(?m)^  qdrant:\s*$", content)
    assert "qdrant/qdrant:v1.18.3-unprivileged" in content
    assert re.search(r"qdrant:\s*\n(?:.*\n){1,3}\s+profiles:\s*\n\s+- local", content)
    assert '"127.0.0.1:6333:6333"' in content
    assert '"127.0.0.1:6334:6334"' in content
    assert "qdrant_storage:/qdrant/storage" in content
    assert re.search(r"(?m)^volumes:\s*\n  qdrant_storage:\s*$", content)
    assert "</dev/tcp/127.0.0.1/6333" in content


def test_container_files_do_not_contain_credential_values():
    content = "\n".join(
        (_read(DOCKERFILE), _read(COMPOSE), _read(PROJECT_ROOT / ".env.example"))
    )

    for forbidden in (
        "AKIA",
        "ASIA",
        "aws_secret_access_key=",
        "aws_session_token=",
        "password=",
    ):
        assert forbidden.casefold() not in content.casefold()


def test_runtime_dependencies_are_bounded_and_development_is_separate():
    runtime = _read(PROJECT_ROOT / "requirements.txt")
    development = _read(PROJECT_ROOT / "requirements-dev.txt")

    for package in (
        "aws-cdk-lib",
        "boto3",
        "constructs",
        "matplotlib",
        "qdrant-client",
        "requests",
        "streamlit",
    ):
        line = next(
            item for item in runtime.splitlines() if item.startswith(package)
        )
        assert ">=" in line
        assert "<" in line
    assert "pytest==8.4.2" in development
    assert "pytest" not in runtime
