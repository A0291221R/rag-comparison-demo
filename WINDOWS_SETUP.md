# Windows Setup Guide

Complete setup instructions for running the RAG Comparison Demo on Windows 10/11.

---

## Prerequisites

### 1. Python 3.11+

Download from https://www.python.org/downloads/windows/

**Critical during install:**
- ✅ Check **"Add Python to PATH"**
- ✅ Check **"Install pip"**

Verify in PowerShell:
```powershell
python --version   # should print Python 3.11.x or higher
pip --version
```

### 2. Docker Desktop

Download from https://www.docker.com/products/docker-desktop/

- Enable **WSL 2 backend** when prompted (recommended for performance)
- After install, open Docker Desktop and wait for the whale icon to stop animating
- Verify:
```powershell
docker --version
docker compose version
```

### 3. Git (optional but recommended)

Download from https://git-scm.com/download/win

---

## Quickstart

Open **PowerShell** (or Windows Terminal) as a regular user (not Administrator).

```powershell
# 1. Clone or extract the project
cd C:\Projects   # or wherever you want it

# If you have git:
git clone https://github.com/yourorg/rag-comparison-demo
cd rag-comparison-demo

# If you downloaded the zip, extract it and cd into it

# 2. Allow PowerShell scripts (one-time, if blocked)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Check prerequisites
.\run.ps1 check-prereqs

# 4. Create your .env file
.\run.ps1 env-setup
notepad .env       # add your OPENAI_API_KEY here

# 5. Start Docker services
.\run.ps1 docker-up

# 6. Install Python packages
.\run.ps1 install

# 7. Ingest sample documents
.\run.ps1 ingest

# 8. Launch the UI
.\run.ps1 run-gradio
# Open http://localhost:7860 in your browser
```

---

## Recommended: uv (Faster Package Manager)

`uv` is dramatically faster than `pip` for installing packages:

```powershell
# Install uv
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart your terminal, then:
.\run.ps1 install-uv
```

---

## Windows-Specific Notes

### PowerShell Execution Policy

If you see *"running scripts is disabled"*:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Long Path Support (recommended)

Some Python packages have long file paths. Enable in PowerShell (as Administrator):
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

Or via Group Policy: Computer Configuration → Administrative Templates → System → Filesystem → Enable Win32 long paths.

### spaCy model download

If the spaCy download fails due to network restrictions:
```powershell
python -m spacy download en_core_web_sm --direct
```

### Line endings (Git)

If you cloned via Git on Windows and see issues, configure:
```powershell
git config --global core.autocrlf input
```

### WSL2 vs Hyper-V for Docker

For best performance use the WSL2 backend (default in Docker Desktop 4.x).
If you must use Hyper-V, enable it in Windows Features:
```powershell
# Run as Administrator
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
```

---

## Service URLs After `.\run.ps1 docker-up`

| Service | URL | Credentials |
|---------|-----|-------------|
| **Gradio UI** | http://localhost:7860 | — |
| **REST API docs** | http://localhost:8000/docs | — |
| **Neo4j Browser** | http://localhost:7474 | neo4j / password |
| **Qdrant Dashboard** | http://localhost:6333/dashboard | — |
| **Grafana** | http://localhost:3000 | admin / admin |
| **Jaeger UI** | http://localhost:16686 | — |

---

## Running Components Separately (Multiple Terminals)

Open three PowerShell windows:

**Terminal 1 — REST API:**
```powershell
cd C:\Projects\rag-comparison-demo
.\run.ps1 run-api
```

**Terminal 2 — Gradio UI:**
```powershell
cd C:\Projects\rag-comparison-demo
.\run.ps1 run-gradio
```

**Terminal 3 — Async Worker (optional):**
```powershell
cd C:\Projects\rag-comparison-demo
.\run.ps1 run-worker
```

---

## Windows Terminal (Recommended)

Install Windows Terminal from the Microsoft Store for a much better experience:
- Multiple tabs for running API, UI, and worker simultaneously
- Better font rendering, copy/paste, colors

---

## Firewall

If prompted by Windows Defender Firewall when starting Docker or Python servers,
click **Allow access** for private networks.

---

## Troubleshooting

### `docker: command not found`
Docker Desktop is not running. Open it from the Start menu and wait for it to initialize.

### `pip install` fails with `Microsoft Visual C++ required`
Install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
Select "Desktop development with C++" workload.

### `ModuleNotFoundError` after install
Make sure you're running in the same Python environment where you installed:
```powershell
where python    # confirm which Python is being used
pip list | findstr langchain
```

### Port already in use
```powershell
# Find what's using port 8000
netstat -ano | findstr :8000
# Kill by PID
taskkill /PID <PID> /F
```

### Neo4j won't start (WSL2 memory)
Add a `.wslconfig` file at `C:\Users\<YourUsername>\.wslconfig`:
```ini
[wsl2]
memory=6GB
processors=4
```
Then restart WSL: `wsl --shutdown`

### `Set-ExecutionPolicy` blocked by organisation policy
Use the `cmd.bat` alternative runner instead of PowerShell scripts:
```cmd
python data\ingest.py
python ui\gradio_app.py
```
