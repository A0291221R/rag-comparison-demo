@echo off
REM run.bat — Windows CMD fallback (for environments where PowerShell scripts are blocked)
REM Usage: run.bat <command>
REM Example: run.bat install
REM          run.bat docker-up
REM          run.bat run-gradio

setlocal
set COMMAND=%1
set PYTHONPATH=%~dp0

if "%COMMAND%"=="" goto help
if "%COMMAND%"=="help" goto help
if "%COMMAND%"=="check-prereqs" goto check_prereqs
if "%COMMAND%"=="env-setup" goto env_setup
if "%COMMAND%"=="install" goto install
if "%COMMAND%"=="docker-up" goto docker_up
if "%COMMAND%"=="docker-down" goto docker_down
if "%COMMAND%"=="docker-logs" goto docker_logs
if "%COMMAND%"=="docker-reset" goto docker_reset
if "%COMMAND%"=="ingest" goto ingest
if "%COMMAND%"=="run-gradio" goto run_gradio
if "%COMMAND%"=="run-api" goto run_api
if "%COMMAND%"=="run-worker" goto run_worker
if "%COMMAND%"=="test" goto test
if "%COMMAND%"=="lint" goto lint
if "%COMMAND%"=="format" goto format
if "%COMMAND%"=="clean" goto clean
if "%COMMAND%"=="proto" goto proto

echo Unknown command: %COMMAND%
goto help

:check_prereqs
echo Checking prerequisites...
python --version || echo [MISSING] Python not found - download from python.org
pip --version || echo [MISSING] pip not found
docker --version || echo [MISSING] Docker not found - install Docker Desktop
echo Done.
goto end

:env_setup
if not exist ".env" (
    copy ".env.example" ".env"
    echo .env created. Open it in Notepad and set your OPENAI_API_KEY.
    notepad .env
) else (
    echo .env already exists.
)
goto end

:install
echo Installing Python dependencies...
pip install -e ".[all]"
python -m spacy download en_core_web_sm
goto end

:docker_up
echo Starting Docker services...
docker compose up -d
timeout /t 10 /nobreak > nul
docker compose ps
echo.
echo Neo4j    -^> http://localhost:7474  (neo4j / password)
echo Qdrant   -^> http://localhost:6333/dashboard
echo Grafana  -^> http://localhost:3000  (admin / admin)
echo Jaeger   -^> http://localhost:16686
goto end

:docker_down
docker compose down
goto end

:docker_logs
docker compose logs -f
goto end

:docker_reset
docker compose down -v
docker compose up -d
goto end

:ingest
echo Ingesting sample corpus...
python data\ingest.py
goto end

:run_gradio
echo Starting Gradio UI on http://localhost:7860
python ui\gradio_app.py
goto end

:run_api
echo Starting FastAPI REST API on http://localhost:8000
uvicorn api.rest.app:app --reload --host 0.0.0.0 --port 8000
goto end

:run_worker
echo Starting Redis Streams worker...
python api\rest\worker.py
goto end

:test
echo Running unit tests...
pytest tests\ -v -m "not integration and not eval"
goto end

:lint
ruff check .
goto end

:format
ruff format .
ruff check --fix .
goto end

:proto
python -m grpc_tools.protoc -I api\grpc --python_out=api\grpc --grpc_python_out=api\grpc api\grpc\rag_service.proto
goto end

:clean
echo Cleaning build artifacts...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul
if exist .pytest_cache rd /s /q .pytest_cache
if exist .ruff_cache rd /s /q .ruff_cache
if exist dist rd /s /q dist
if exist build rd /s /q build
echo Clean complete.
goto end

:help
echo.
echo RAG Comparison Demo — Windows CMD Commands
echo ==========================================
echo.
echo   run.bat check-prereqs    Check Python, Docker installed
echo   run.bat env-setup        Copy .env.example to .env
echo   run.bat install          Install Python packages
echo   run.bat docker-up        Start all Docker services
echo   run.bat docker-down      Stop all Docker services
echo   run.bat docker-reset     Wipe volumes and restart
echo   run.bat docker-logs      Tail service logs
echo   run.bat ingest           Load sample corpus
echo   run.bat run-gradio       Start UI at http://localhost:7860
echo   run.bat run-api          Start API at http://localhost:8000
echo   run.bat run-worker       Start Redis async worker
echo   run.bat test             Run unit tests
echo   run.bat lint             Lint with ruff
echo   run.bat format           Format with ruff
echo   run.bat proto            Generate gRPC stubs
echo   run.bat clean            Remove __pycache__ and build files
echo.
echo Quickstart:
echo   1. run.bat check-prereqs
echo   2. run.bat env-setup        ^(edit .env with your API keys^)
echo   3. run.bat docker-up
echo   4. run.bat install
echo   5. run.bat ingest
echo   6. run.bat run-gradio       ^(visit http://localhost:7860^)
echo.

:end
endlocal
