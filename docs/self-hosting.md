# Self-hosting on an always-on machine

Run the dashboard, API, and chat assistant on a small always-on computer on your home network, such as a Raspberry Pi, a mini PC, or an old laptop running Linux. Predictions refresh on their own every day, so you can check them from any device on your Wi-Fi without keeping your main computer on.

> **Raspberry Pi notes** are called out in boxes like this one. Everything else applies to any 64-bit Linux machine.

## What runs where

| Piece | Runs | Notes |
|---|---|---|
| `api` (FastAPI) | always | Serves predictions on port 8000 |
| `app` (Streamlit + chat) | always | The dashboard on port 8501 |
| `job` (prediction refresh) | daily, scheduled | About a minute: pulls new data and updates predictions |
| `bootstrap` (data + training) | once, optional | Or train on your main computer and copy the results over (step 5) |

## Requirements

- 64-bit Linux (`arm64` or `x86_64`), **4 GB RAM** or more, and about **5 GB free disk**.
- Docker Engine with the Compose plugin.
- An Anthropic API key for the chat (optional; the dashboard works without it).

> **Raspberry Pi:** use a Pi 4 or 5 with **64-bit Raspberry Pi OS** (Bookworm or newer). `uname -m` must print `aarch64`; the 32-bit OS won't work. A USB SSD is more reliable than a microSD card for an always-on machine. Every dependency has a prebuilt ARM64 package, so nothing compiles from source during the build.

## 1. Install Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh                    # Docker's official install script; read it first if you like
sudo usermod -aG docker $USER       # use docker without sudo; log out and back in afterwards
docker compose version              # check that the Compose plugin is installed
```

## 2. Get the code

```bash
git clone https://github.com/Programming-Mellow/vikings-win-predictor.git
cd vikings-win-predictor
```

If the repository is private, sign in first: install the GitHub CLI and run `gh auth login`, or clone with a [personal access token](https://github.com/settings/tokens).

## 3. Add your API key

Create the data folders first, so they're owned by your user and not by root. The container runs as a regular user and needs to write to them.

```bash
mkdir -p data models mlruns
cp .env.example .env
nano .env                           # set ANTHROPIC_API_KEY=...
chmod 600 .env                      # only your user can read it
```

Tip: set a monthly spending limit in the [Anthropic Console](https://console.anthropic.com/) so the chat can never cost more than you expect.

## 4. Build the image

```bash
docker compose build
```

The first build takes a few minutes on a desktop CPU.

> **Raspberry Pi 4:** expect 10–20 minutes. If the build runs out of memory, add swap: in `/etc/dphys-swapfile` set `CONF_SWAPSIZE=2048`, then run `sudo systemctl restart dphys-swapfile`.

Then check that the compiled libraries run on your CPU:

```bash
docker compose run --rm api python -c "import polars, xgboost, shap, pyarrow; print('ok')"
```

If this prints `Illegal instruction` instead of `ok`, see [Troubleshooting](#troubleshooting).

## 5. Add data and models

Pick one option.

**Option A: copy from the computer you already trained on (recommended for a Pi 4).** From that computer, in the project folder:

```bash
scp -r data models reports mlruns <user>@<host>.local:~/vikings-win-predictor/
```

`scp` is built into Windows PowerShell, macOS, and Linux. Replace `<user>` and `<host>` with the server's login and hostname (run `hostname` on the server to see it).

**Option B: build everything on the server.**

```bash
docker compose --profile bootstrap run --rm bootstrap
```

This downloads 2010–present data, trains both models, and predicts the current season. It takes about 2 minutes on a desktop and roughly 15–30 minutes on a Pi 4.

## 6. Start it

```bash
docker compose up -d
docker compose ps                   # api should show "healthy" and app "running"
```

Open it from any device on your home Wi-Fi:

- Dashboard and chat: `http://<host>.local:8501`
- API docs: `http://<host>.local:8000/docs`

If `<host>.local` doesn't resolve on your network, use the server's IP address instead (`hostname -I` prints it). Both services restart on their own after a reboot or power cut.

## 7. Refresh predictions every day

Add a cron job that runs the refresh each morning:

```bash
crontab -e
```

Add this line (it runs at 9:00 every day in the server's time zone):

```cron
0 9 * * * cd $HOME/vikings-win-predictor && docker compose --profile job run --rm job >> $HOME/vikings-job.log 2>&1
```

Why daily: betting lines and injury reports change during the week, and a morning run locks in a fresh pre-game prediction before Sunday kickoffs. A daily run also picks up Thursday and Monday night games. Check the time zone with `timedatectl`.

> **Raspberry Pi:** set the time zone with `sudo raspi-config` → *Localisation Options* → *Timezone*.

To run a refresh right away, use the same command without the schedule:

```bash
docker compose --profile job run --rm job
```

## Updating

```bash
git pull
docker compose build
docker compose up -d
```

Your data, models, predictions, and learner profile live in the mounted folders, so updates keep them. If you trained on the server (option B), training rewrote the charts in `reports/`. If `git pull` complains about them, run `git checkout -- reports` first.

## Day-to-day

| Task | Command |
|---|---|
| See what's running | `docker compose ps` |
| Follow app logs | `docker compose logs -f app` |
| Check the last refresh | `tail -n 20 ~/vikings-job.log` |
| Stop everything | `docker compose down` |
| Free disk from old images | `docker image prune` |

## Security

- **Keep it on your home network.** Don't forward ports 8000 or 8501 on your router. The app has no login, so anyone who reached it could use the chat on your API key.
- `.env` holds your API key: keep it out of Git (it already is) and readable only by you (`chmod 600 .env`).
- To use it away from home later, put a private network such as Tailscale or WireGuard in front of it instead of exposing ports.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Illegal instruction` in the step 4 check | A compiled library doesn't support this CPU. Note which import fails (run them one at a time) and pin an older version of that package in `pyproject.toml`, or train on another computer (step 5, option A). Only the import that fails needs a fix. |
| `Permission denied` writing to `data/` or `models/` | Docker created the folders as root. Fix ownership: `sudo chown -R 1000:1000 data models reports mlruns`. |
| Build killed or very slow | Out of memory: add swap (see step 4), or close other programs during the build. |
| Dashboard says it can't reach the API | `docker compose ps`: if `api` isn't healthy, check `docker compose logs api`. Usually the `models/` or `data/` folder is missing (step 5). |
| "No predictions for 2026" | Run the refresh once: `docker compose --profile job run --rm job`. |
| Chat says to add a key | `.env` is missing or the key is empty; fix it, then `docker compose up -d`. |
| Port already in use | Another program is using 8000 or 8501. Change the left-hand number in `ports:` in `docker-compose.yml`, e.g. `"8080:8501"`. |
