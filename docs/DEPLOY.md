# Deploying Invoice Guard

## What the server needs
- Linux x86-64 with Docker Engine and the Compose plugin.
- About 2 GB of free disk for the image, and 3 GB of memory for the container.
- Network access to the Ollama server, only if field extraction is used.

## 1. Get the code onto the server

**Option A: build on the server** (needs internet access during the build). Copy the project
folder without `.venv`, `.env`, `test-images` and `out`.

**Option B: copy a built image** (server needs no internet).

```bash
# on the build machine
docker save invoice-guard:0.1.0 -o invoice-guard-0.1.0.tar
# copy the .tar, docker-compose.yml and .env.example to the server, then there:
docker load -i invoice-guard-0.1.0.tar
```

## 2. Configure
Create `.env` next to `docker-compose.yml`:

```
# 127.0.0.1 = this server only. 0.0.0.0 = reachable from the network (read "Security" first).
IG_BIND=127.0.0.1
IG_PORT=8000

# Field extraction. Set IG_ENABLE_EXTRACTION=false if there is no Ollama server.
IG_ENABLE_EXTRACTION=true
IG_OLLAMA_URL=http://YOUR-OLLAMA-HOST:11434
IG_OLLAMA_MODEL=qwen2.5vl:7b
IG_OLLAMA_FALLBACK_MODEL=qwen2.5vl:3b
```

Inside a container `localhost` is the container itself. If Ollama runs on the same server, use
`http://host.docker.internal:11434`; Ollama must then listen on more than `127.0.0.1`.

## 3. Run

```bash
docker compose up -d --build     # Option A
docker compose up -d --no-build  # Option B
docker compose ps                # wait for "healthy"
curl http://127.0.0.1:8000/healthz
```

Open `http://SERVER:8000/` (or the port you chose).

## Operate

```bash
docker compose logs -f           # logs
docker compose restart           # restart
docker compose down              # stop (the data volume is kept)
```

The only stored data is the duplicate-check database in the `invoice-guard_ig-data` volume.
Uploaded files are processed in memory and are not saved.

## Security: read before exposing it
- **There is no login.** Anyone who can reach the port can upload files and use the model server.
- Keep `IG_BIND=127.0.0.1` and put a reverse proxy in front that provides **authentication and
  HTTPS**, or restrict the port with a firewall to trusted addresses.
- Limit upload size at the proxy to match `IG_MAX_UPLOAD_MB` (25 by default).
- Page images are sent to the Ollama server over the URL you configure; use a trusted network.
- The container runs as a non-root user with a read-only file system and no extra privileges.
