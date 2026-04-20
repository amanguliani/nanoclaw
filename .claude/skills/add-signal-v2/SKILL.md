# Add Signal Channel (v2)

Adds Signal messaging support to NanoClaw v2 using `signal-sdk` (a TypeScript
wrapper around `signal-cli`). Unlike Telegram/Discord, Signal has no bot API —
NanoClaw registers as a full Signal account on a dedicated phone number.

**Two registration paths:**
- **New number (recommended):** Register a dedicated SIM or VoIP number as a
  standalone Signal account. NanoClaw owns the number entirely.
- **Linked device:** Join an existing Signal account as a secondary device via
  QR code. Simpler, but NanoClaw shares your personal number.

Both paths are documented below. The new-number path is battle-tested.

---

## Pre-flight

Check if `src/channels/signal.ts` exists and the import is uncommented in
`src/channels/index.ts`. If both are in place, skip to Registration.

## Install

### 1. Check Java

Java 17+ is required. Check:

```bash
java -version
```

If missing:
- **RHEL/CentOS/Fedora:** `sudo dnf install -y java-17-openjdk`
- **Debian/Ubuntu:** `sudo apt-get install -y default-jre`
- **macOS:** `brew install --cask temurin@17`

Java 17–25 all work. Java 25 (RHEL9 default) is confirmed working.

### 2. Install signal-cli

The `signal-sdk` npm package bundles signal-cli, but the bundled version is
often outdated. Install the latest standalone binary:

```bash
SIGNAL_CLI_VERSION=$(curl -fsSL https://api.github.com/repos/AsamK/signal-cli/releases/latest | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'][1:])")
curl -fsSL "https://github.com/AsamK/signal-cli/releases/download/v${SIGNAL_CLI_VERSION}/signal-cli-${SIGNAL_CLI_VERSION}-Linux-native.tar.gz" \
  | tar -xz -C ~/.local
ln -sf ~/.local/signal-cli ~/.local/bin/signal-cli
signal-cli --version
```

> **Note:** The Linux native tarball extracts a single binary directly to
> `~/.local/signal-cli` (not into a subdirectory). The symlink above handles this.

### 3. Install signal-sdk

```bash
npm install signal-sdk
```

### 4. Enable the adapter

Uncomment the Signal import in `src/channels/index.ts`:

```typescript
import './signal.js';
```

### 5. Build

Always build with Node 22 (nvm):

```bash
source ~/.nvm/nvm.sh && nvm use 22 && npm run build
```

---

## Path A: Register a new Signal number

Use this if you have a dedicated SIM or VoIP number for NanoClaw.

> **VoIP numbers:** Signal requires SMS verification before voice. Some VoIP
> providers are blocked even for voice calls. If registration fails with an auth
> error, try a different provider or a physical SIM.

### Step 1: Request SMS verification

Signal requires a CAPTCHA on first registration:

1. Open `https://signalcaptchas.org/registration/generate.html` in a browser
2. Solve the captcha
3. Right-click the **"Open Signal"** button → **Copy Link**
4. The link starts with `signalcaptcha://...`

```bash
SIGNAL_CLI_CONFIG_PATH=/path/to/nanoclaw/data/signal \
  signal-cli -u +YOURNUMBER register \
  --captcha "PASTE_CAPTCHA_TOKEN_HERE"
```

The captcha token is everything after `signalcaptcha://` in the copied link.

### Step 2: Voice call fallback (VoIP numbers without SMS)

If your number cannot receive SMS, wait ~60 seconds after the SMS request then
request a voice call:

```bash
SIGNAL_CLI_CONFIG_PATH=/path/to/nanoclaw/data/signal \
  signal-cli -u +YOURNUMBER register --voice \
  --captcha "SAME_CAPTCHA_TOKEN"
```

Signal will call your number and read a 6-digit code.

> The captcha token from Step 1 is reusable for the voice retry — no need to
> solve a new one.

### Step 3: Verify

```bash
SIGNAL_CLI_CONFIG_PATH=/path/to/nanoclaw/data/signal \
  signal-cli -u +YOURNUMBER verify CODE
```

No output = success.

### Step 4: Set profile name (optional)

> ⚠ signal-sdk holds an exclusive lock on `data/signal/` while nanoclaw is
> running. Stop the service before running signal-cli commands, then restart.

```bash
systemctl --user stop nanoclaw
SIGNAL_CLI_CONFIG_PATH=/path/to/nanoclaw/data/signal \
  signal-cli -u +YOURNUMBER updateProfile --name "YourBotName"
systemctl --user start nanoclaw
```

To set an avatar too:
```bash
signal-cli -u +YOURNUMBER updateProfile --name "YourBotName" --avatar /path/to/avatar.jpg
```

---

## Path B: Link as secondary device

Use this to join an existing Signal account as a secondary device.

```bash
mkdir -p data/signal
export SIGNAL_CLI_CONFIG_PATH=$(pwd)/data/signal
npx signal-sdk link -n "NanoClaw" -a +YOURNUMBER
```

This prints a QR code. On your phone: **Settings → Linked Devices → Link New Device**.
Scan the code within ~30 seconds.

---

## Configure environment

Add to `.env`:

```bash
SIGNAL_PHONE_NUMBER=+YOURNUMBER
```

---

## Wire to an agent

### DMs

After the service starts, **send any message** to the Signal number from your
personal Signal app. The router auto-creates a `messaging_groups` row. Then:

```bash
sqlite3 data/v2.db \
  "SELECT id, platform_id FROM messaging_groups WHERE channel_type='signal' ORDER BY created_at DESC LIMIT 5"
```

Pass the `id` and `platform_id` to `/init-first-agent` or wire manually.

**Important:** DM `platform_id` is UUID-based, not phone-based:
- DM: `signal:3de71d7f-ffa3-437e-b4db-097534bccd46`  (UUID of sender)
- Group: `signal:UwaIz6bc09Olg1pBr/XeBuQf6z3fyCZoNj/Y3Tpe3hI=`  (base64 group ID)

### Groups

Add the Signal number to a group from your phone. Send any message — the router
auto-creates the group's `messaging_groups` row. Wire it:

```bash
sqlite3 data/v2.db \
  "SELECT id, platform_id FROM messaging_groups WHERE channel_type='signal' ORDER BY created_at DESC LIMIT 5"
```

Then insert the wiring (replace IDs):

```bash
NOW=$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")
sqlite3 data/v2.db "
INSERT OR IGNORE INTO messaging_group_agents
  (id, messaging_group_id, agent_group_id, session_mode, priority, created_at)
VALUES
  ('mga-'||hex(randomblob(8)), 'mg-GROUPID', 'ag-AGENTID', 'isolated', 0, '$NOW');
"
```

Use `session_mode='isolated'` for groups so each group has its own session.

### Grant user access

Users who message via Signal need to be granted membership. Without this,
messages are silently dropped with `not_member`. After first contact:

```bash
NOW=$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")
sqlite3 data/v2.db "
INSERT OR REPLACE INTO user_roles (user_id, role, agent_group_id, granted_by, granted_at)
  VALUES ('signal:UUID', 'owner', NULL, 'system', '$NOW');
INSERT OR IGNORE INTO agent_group_members (user_id, agent_group_id, added_by, added_at)
  VALUES ('signal:UUID', 'ag-AGENTID', 'system', '$NOW');
"
```

Find the UUID from `messaging_groups.platform_id` or the `users` table.

---

## Voice Transcription (optional)

Inbound voice messages are automatically transcribed using a local whisper.cpp
binary. The feature degrades gracefully — if whisper-cli or ffmpeg is missing,
voice messages are still delivered as attachments with no transcript.

### Install whisper.cpp on Linux

```bash
# Build from source (requires git, cmake, make, gcc)
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build && cmake --build build --config Release -j$(nproc)
sudo cp build/bin/whisper-cli /usr/local/bin/whisper-cli
```

Or install ffmpeg + Python openai-whisper (slower but easier):
```bash
sudo dnf install -y ffmpeg        # or: sudo apt install ffmpeg
pip3 install openai-whisper
# then set WHISPER_BIN=whisper and WHISPER_MODEL=base in .env
```

On macOS: `brew install whisper-cpp ffmpeg`

### Download a model

```bash
mkdir -p data/models
curl -L -o data/models/ggml-base.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
```

Larger models trade speed for accuracy: `ggml-small.bin` (466 MB), `ggml-medium.bin` (1.5 GB).

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_BIN` | `whisper-cli` | Path to whisper.cpp binary |
| `WHISPER_MODEL` | `data/models/ggml-base.bin` | Path to GGML model file |

### Verify

```bash
ffmpeg -version >/dev/null && echo "ffmpeg OK" || echo "ffmpeg missing"
whisper-cli --version 2>/dev/null && echo "whisper-cli OK" || echo "whisper-cli missing"
ls data/models/ggml-*.bin 2>/dev/null || echo "no model — download one"
```

---

## Channel Info

- **type**: `signal`
- **terminology**: "Chats" (1:1) and "groups" (multi-member). No threads.
- **supports-threads**: no
- **inbound**: text, reactions (forwarded to agent with emoji + targetTimestamp), images, files, voice (transcribed via whisper-cli if installed)
- **outbound**: text, reactions (sendReaction), file attachments
- **platform-id-format**:
  - DM: `signal:{UUID}` — sender's Signal UUID, **not** their phone number
  - Group: `signal:{base64GroupId}`
- **how-to-find-id**: Send a message to the bot, then query `messaging_groups`
  as shown above.
- **config-lock**: signal-sdk holds an exclusive lock on `data/signal/` while
  nanoclaw is running. Stop the service before running any `signal-cli` commands.

---

## Troubleshooting

**`Config file is in use by another instance`** — nanoclaw is running and
signal-sdk has the lock. Stop the service, run the command, restart.

**Messages dropped with `not_member`** — the Signal user hasn't been granted
membership. See "Grant user access" above. This affects every new Signal user,
including the owner's Signal identity (which is separate from their Telegram
identity even if it's the same person).

**Captcha required** — Signal requires captcha for new registrations. Go to
`https://signalcaptchas.org/registration/generate.html`, solve it, right-click
"Open Signal", copy the link, extract the token after `signalcaptcha://`.

**`Invalid verification method: Before requesting voice verification…`** —
You must request SMS first, wait ~60 seconds, then request voice. Both can use
the same captcha token.

**`The provided model identifier is invalid` (Bedrock)** — unrelated to Signal;
this is a LiteLLM model ID issue.

**Group replies going to DM instead of group** — modern Signal groups use
GroupV2. The adapter must extract the group ID from
`envelope?.dataMessage?.groupV2?.id` (not just `groupInfo?.groupId`, which is
GroupV1/legacy). Check `src/channels/signal.ts` and confirm the groupId
extraction falls through to `groupV2.id`.

**Java not found** — install Java 17+ (see Install step 1).

**QR code expired (Path B)** — QR codes expire in ~30 seconds. Re-run the
link command to generate a new one.

**signal-cli binary location** — The native Linux tarball extracts directly to
`~/.local/signal-cli` (a single file, not a directory). The system `aws` or
other tools named `signal-cli` won't be in PATH by default; check
`~/.local/bin/signal-cli`.
