# run.ps1 — Windows PowerShell equivalent of the Makefile
# Run from project root: .\run.ps1 <command>
# Example: .\run.ps1 install
#          .\run.ps1 docker-up
#          .\run.ps1 run-gradio

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

function Write-Header($msg) {
    Write-Host "`n==> $msg" -ForegroundColor Cyan
}

function Check-Command($cmd) {
    return [bool](Get-Command $cmd -ErrorAction SilentlyContinue)
}

switch ($Command) {

    "install" {
        Write-Header "Installing Python dependencies"
        if (Check-Command "uv") {
            uv sync --all-extras
        } else {
            pip install -e ".[all]"
        }
        Write-Header "Downloading spaCy model"
        python -m spacy download en_core_web_sm
    }

    "install-uv" {
        Write-Header "Installing uv package manager"
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
        uv sync --all-extras
    }

    "docker-up" {
        Write-Header "Starting Docker services (Neo4j, Qdrant, Redis, Postgres, OTel, Jaeger, Grafana)"
        docker compose up -d
        Write-Host "Waiting for services to be healthy..." -ForegroundColor Yellow
        Start-Sleep -Seconds 10
        docker compose ps
        Write-Host "`nServices ready:" -ForegroundColor Green
        Write-Host "  Neo4j    -> http://localhost:7474  (neo4j / password)"
        Write-Host "  Qdrant   -> http://localhost:6333/dashboard"
        Write-Host "  Grafana  -> http://localhost:3000  (admin / admin)"
        Write-Host "  Jaeger   -> http://localhost:16686"
        Write-Host "  Redis    -> localhost:6379"
    }

    "docker-down" {
        Write-Header "Stopping Docker services"
        docker compose down
    }

    "docker-reset" {
        Write-Header "Resetting Docker volumes"
        docker compose down -v
        docker compose up -d
    }

    "docker-logs" {
        docker compose logs -f
    }

    "ingest" {
        Write-Header "Ingesting sample corpus into Qdrant + Neo4j"
        python data/ingest.py
    }

    "build-graph" {
        Write-Header "Building Neo4j knowledge graph schema"
        python graph_db/schema/build_graph.py
    }

    "run-gradio" {
        Write-Header "Starting Gradio UI on http://localhost:7860"
        $env:PYTHONPATH = $ProjectRoot
        python ui/gradio_app.py
    }

    "run-api" {
        Write-Header "Starting FastAPI REST API on http://localhost:8000"
        $env:PYTHONPATH = $ProjectRoot
        $port = if ($env:API_PORT) { $env:API_PORT } else { "8000" }
        uvicorn api.rest.app:app --reload --host 0.0.0.0 --port $port
    }

    "run-worker" {
        Write-Header "Starting Redis Streams worker"
        $env:PYTHONPATH = $ProjectRoot
        python api/rest/worker.py
    }

    "run-grpc" {
        Write-Header "Starting gRPC server on port 50051"
        $env:PYTHONPATH = $ProjectRoot
        python api/grpc/server.py
    }

    "test" {
        Write-Header "Running unit tests"
        $env:PYTHONPATH = $ProjectRoot
        pytest tests/ -v -m "not integration and not eval"
    }

    "test-integration" {
        Write-Header "Running integration tests (requires live services)"
        $env:PYTHONPATH = $ProjectRoot
        pytest tests/test_integration.py -v -m integration
    }

    "lint" {
        Write-Header "Linting with ruff"
        ruff check .
    }

    "format" {
        Write-Header "Formatting with ruff"
        ruff format .
        ruff check --fix .
    }

    "proto" {
        Write-Header "Generating gRPC stubs from .proto"
        python -m grpc_tools.protoc `
            -I api/grpc `
            --python_out=api/grpc `
            --grpc_python_out=api/grpc `
            api/grpc/rag_service.proto
    }

    "finetune" {
        Write-Header "Fine-tuning embedding model"
        $env:PYTHONPATH = $ProjectRoot
        python finetuning/dataset_prep.py
        python finetuning/train.py
        python finetuning/eval.py
    }

    "clean" {
        Write-Header "Cleaning build artifacts"
        Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
        Get-ChildItem -Path . -Recurse -Filter "*.pyc" | Remove-Item -Force
        Get-ChildItem -Path . -Recurse -Filter "*.pyo" | Remove-Item -Force
        @(".pytest_cache", ".ruff_cache", ".mypy_cache", "dist", "build") | ForEach-Object {
            if (Test-Path $_) { Remove-Item $_ -Recurse -Force }
        }
        Write-Host "Clean complete." -ForegroundColor Green
    }

    "env-setup" {
        Write-Header "Creating .env from .env.example"
        if (-not (Test-Path ".env")) {
            Copy-Item ".env.example" ".env"
            Write-Host ".env created. Edit it and set your API keys." -ForegroundColor Yellow
        } else {
            Write-Host ".env already exists — skipping." -ForegroundColor Yellow
        }
    }

    "check-prereqs" {
        Write-Header "Checking prerequisites"
        $ok = $true
        foreach ($tool in @("python", "docker", "pip")) {
            if (Check-Command $tool) {
                $ver = & $tool --version 2>&1
                Write-Host "  [OK] $tool : $ver" -ForegroundColor Green
            } else {
                Write-Host "  [MISSING] $tool" -ForegroundColor Red
                $ok = $false
            }
        }
        if (Check-Command "uv") {
            Write-Host "  [OK] uv (fast package manager available)" -ForegroundColor Green
        } else {
            Write-Host "  [INFO] uv not found — will use pip instead" -ForegroundColor Yellow
        }
        if ($ok) {
            Write-Host "`nAll required tools found." -ForegroundColor Green
        } else {
            Write-Host "`nSome tools are missing. See WINDOWS_SETUP.md" -ForegroundColor Red
        }
    }

    "help" {
        Write-Host @"

RAG Comparison Demo — Windows PowerShell Commands
==================================================

Setup:
  .\run.ps1 check-prereqs     Check Python, Docker, pip are installed
  .\run.ps1 env-setup         Copy .env.example -> .env (then add your API keys)
  .\run.ps1 install           Install Python dependencies
  .\run.ps1 install-uv        Install uv (fast) then install deps

Docker:
  .\run.ps1 docker-up         Start all services (Neo4j, Qdrant, Redis, etc.)
  .\run.ps1 docker-down       Stop all services
  .\run.ps1 docker-reset      Wipe volumes and restart
  .\run.ps1 docker-logs       Tail all service logs

Data:
  .\run.ps1 ingest            Load sample corpus into Qdrant + Neo4j

Run:
  .\run.ps1 run-gradio        Start UI at http://localhost:7860
  .\run.ps1 run-api           Start REST API at http://localhost:8000/docs
  .\run.ps1 run-worker        Start Redis Streams async worker
  .\run.ps1 run-grpc          Start gRPC server

Dev:
  .\run.ps1 test              Run unit tests
  .\run.ps1 test-integration  Run integration tests (needs live services)
  .\run.ps1 lint              Lint with ruff
  .\run.ps1 format            Auto-format with ruff
  .\run.ps1 proto             Generate gRPC Python stubs
  .\run.ps1 finetune          Prepare dataset, train, and eval embedding model
  .\run.ps1 clean             Remove __pycache__, .pyc, build artifacts

Quickstart:
  1. .\run.ps1 check-prereqs
  2. .\run.ps1 env-setup          # then edit .env with your API keys
  3. .\run.ps1 docker-up
  4. .\run.ps1 install
  5. .\run.ps1 ingest
  6. .\run.ps1 run-gradio         # visit http://localhost:7860
"@
    }

    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host "Run: .\run.ps1 help" -ForegroundColor Yellow
    }
}
