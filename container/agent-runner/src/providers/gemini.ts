import fs from 'fs';
import path from 'path';

import { GoogleGenAI } from '@google/genai';

import { registerProvider } from './provider-registry.js';
import type { AgentProvider, AgentQuery, Attachment, ProviderEvent, ProviderOptions, QueryInput } from './types.js';

function log(msg: string): void {
  console.error(`[gemini-provider] ${msg}`);
}

const WORKSPACE = '/workspace';
const MAX_TOOL_ROUNDS = 10;

// History is stored per-session in the agent workspace so conversations
// persist across container restarts.
const HISTORY_FILE = path.join(WORKSPACE, 'agent', 'gemini-history.json');

type GeminiContent = { role: string; parts: GeminiPart[] };
type GeminiPart =
  | { text: string }
  | { inlineData: { mimeType: string; data: string } }
  | { functionCall: { name: string; args: unknown } }
  | { functionResponse: { name: string; response: { output: string } } };

const GEMINI_IMAGE_TYPES: Record<string, string> = {
  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.png': 'image/png', '.gif': 'image/gif', '.webp': 'image/webp',
};

function buildGeminiImageParts(attachments: Attachment[]): GeminiPart[] {
  const parts: GeminiPart[] = [];
  for (const a of attachments) {
    if (a.type !== 'image') { log(`skip non-image attachment: ${a.type}`); continue; }
    const fullPath = path.join('/workspace', a.localPath);
    if (!fs.existsSync(fullPath)) { log(`image file not found: ${fullPath}`); continue; }
    const ext = path.extname(a.name).toLowerCase();
    const mimeType = GEMINI_IMAGE_TYPES[ext];
    if (!mimeType) { log(`unknown image ext: ${ext}`); continue; }
    const data = fs.readFileSync(fullPath).toString('base64');
    log(`image loaded: ${fullPath} ext=${ext} mimeType=${mimeType} base64Bytes=${data.length}`);
    parts.push({ inlineData: { mimeType, data } });
  }
  log(`buildGeminiImageParts: ${parts.length} part(s) from ${attachments.length} attachment(s)`);
  return parts;
}

// ---- Google Photos tools -------------------------------------------------------
// Token is read from the mounted stub file; OneCLI intercepts the outbound
// request to photoslibrary.googleapis.com and injects the real OAuth bearer.
const PHOTOS_TOKEN_PATH = '/workspace/extra/.google-photos-mcp/token.json';

function getPhotosToken(): string {
  try {
    return JSON.parse(fs.readFileSync(PHOTOS_TOKEN_PATH, 'utf-8')).access_token || 'onecli-managed';
  } catch {
    return 'onecli-managed';
  }
}

async function photosApiRequest(path_: string, method = 'GET', body?: unknown): Promise<unknown> {
  const resp = await fetch(`https://photoslibrary.googleapis.com${path_}`, {
    method,
    headers: {
      Authorization: `Bearer ${getPhotosToken()}`,
      'Content-Type': 'application/json',
    },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  return resp.json();
}

async function toolListAlbums(args: Record<string, unknown>): Promise<string> {
  const ps = Math.min(Number(args.pageSize) || 20, 50);
  const q = `pageSize=${ps}${args.pageToken ? `&pageToken=${encodeURIComponent(String(args.pageToken))}` : ''}`;
  return JSON.stringify(await photosApiRequest(`/v1/albums?${q}`), null, 2);
}

async function toolGetRecentPhotos(args: Record<string, unknown>): Promise<string> {
  const ps = Math.min(Number(args.pageSize) || 25, 100);
  const q = `pageSize=${ps}${args.pageToken ? `&pageToken=${encodeURIComponent(String(args.pageToken))}` : ''}`;
  return JSON.stringify(await photosApiRequest(`/v1/mediaItems?${q}`), null, 2);
}

async function toolGetAlbumPhotos(args: Record<string, unknown>): Promise<string> {
  const body: Record<string, unknown> = {
    albumId: args.albumId,
    pageSize: Math.min(Number(args.pageSize) || 25, 100),
  };
  if (args.pageToken) body.pageToken = args.pageToken;
  return JSON.stringify(await photosApiRequest('/v1/mediaItems:search', 'POST', body), null, 2);
}

async function toolSearchPhotos(args: Record<string, unknown>): Promise<string> {
  const filters: Record<string, unknown> = {};
  if (args.startDate || args.endDate) {
    const parseDate = (s: unknown) => {
      if (!s) return null;
      const [y, m, d] = String(s).split('-').map(Number);
      return { year: y || 0, month: m || 0, day: d || 0 };
    };
    filters.dateFilter = { ranges: [{ startDate: parseDate(args.startDate), endDate: parseDate(args.endDate) }] };
  }
  if (Array.isArray(args.contentCategories) && args.contentCategories.length > 0) {
    filters.contentFilter = { includedContentCategories: args.contentCategories };
  }
  const body: Record<string, unknown> = { pageSize: Math.min(Number(args.pageSize) || 25, 100), filters };
  if (args.pageToken) body.pageToken = args.pageToken;
  return JSON.stringify(await photosApiRequest('/v1/mediaItems:search', 'POST', body), null, 2);
}

function loadHistory(): GeminiContent[] {
  try {
    if (fs.existsSync(HISTORY_FILE)) {
      return JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf-8'));
    }
  } catch {
    /* start fresh */
  }
  return [];
}

function saveHistory(history: GeminiContent[]): void {
  try {
    fs.mkdirSync(path.dirname(HISTORY_FILE), { recursive: true });
    // Keep last 80 turns to bound token growth.
    // Strip inlineData before persisting — base64 blobs are large, only valid
    // for the turn they were sent, and re-sending them in every future request
    // wastes context and tokens. Replace with a text note so history is coherent.
    const trimmed = history.slice(-80).map((turn) => ({
      ...turn,
      parts: turn.parts.map((p) =>
        'inlineData' in p ? { text: `[image: ${(p as { inlineData: { mimeType: string } }).inlineData.mimeType}]` } : p
      ),
    }));
    fs.writeFileSync(HISTORY_FILE, JSON.stringify(trimmed));
  } catch (err) {
    log(`Failed to save history: ${err}`);
  }
}

// ---- File tools -------------------------------------------------------
// Path-traversal safe: strip leading slash and collapse ".." so all paths
// resolve inside /workspace.
function safeJoin(base: string, userPath: string): string {
  const stripped = (userPath as string).replace(/^\/+/, '');
  const joined = path.join(base, stripped);
  // Ensure resolved path stays inside base
  if (!joined.startsWith(base)) return path.join(base, 'agent', 'forbidden');
  return joined;
}

function toolReadFile(args: Record<string, unknown>): string {
  try {
    return fs.readFileSync(safeJoin(WORKSPACE, args.path as string), 'utf-8');
  } catch (err) {
    return `Error: ${err instanceof Error ? err.message : String(err)}`;
  }
}

function toolWriteFile(args: Record<string, unknown>): string {
  try {
    const dest = safeJoin(WORKSPACE, args.path as string);
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    fs.writeFileSync(dest, args.content as string);
    return `OK: wrote ${args.path}`;
  } catch (err) {
    return `Error: ${err instanceof Error ? err.message : String(err)}`;
  }
}

function toolListFiles(args: Record<string, unknown>): string {
  try {
    const dir = safeJoin(WORKSPACE, (args.path as string) || 'agent');
    return fs.readdirSync(dir).join('\n') || '(empty)';
  } catch (err) {
    return `Error: ${err instanceof Error ? err.message : String(err)}`;
  }
}

async function executeTool(name: string, args: Record<string, unknown>): Promise<string> {
  switch (name) {
    case 'read_file':          return toolReadFile(args);
    case 'write_file':         return toolWriteFile(args);
    case 'list_files':         return toolListFiles(args);
    case 'list_albums':        return toolListAlbums(args);
    case 'get_recent_photos':  return toolGetRecentPhotos(args);
    case 'get_album_photos':   return toolGetAlbumPhotos(args);
    case 'search_photos':      return toolSearchPhotos(args);
    default:                   return `Unknown tool: ${name}`;
  }
}

const TOOLS = [
  {
    functionDeclarations: [
      {
        name: 'read_file',
        description:
          'Read a file from the workspace. ' +
          'Use "extra/fitty-data/workouts/YYYY-MM-DD.md" for workout logs, ' +
          '"extra/fitty-data/photos/" for photos, ' +
          '"agent/" for your own notes.',
        parameters: {
          type: 'object',
          properties: { path: { type: 'string', description: 'Path relative to /workspace' } },
          required: ['path'],
        },
      },
      {
        name: 'write_file',
        description:
          'Write (or append) content to a file in the workspace. ' +
          'Creates parent directories automatically. ' +
          'For workout logs, write to "extra/fitty-data/workouts/YYYY-MM-DD.md".',
        parameters: {
          type: 'object',
          properties: {
            path: { type: 'string', description: 'Path relative to /workspace' },
            content: { type: 'string', description: 'Full file content to write' },
          },
          required: ['path', 'content'],
        },
      },
      {
        name: 'list_files',
        description: 'List files in a workspace directory.',
        parameters: {
          type: 'object',
          properties: { path: { type: 'string', description: 'Directory path relative to /workspace' } },
          required: ['path'],
        },
      },
      {
        name: 'list_albums',
        description: 'List all Google Photos albums. Returns album IDs, titles, and item counts. Only available if Google Photos is connected.',
        parameters: {
          type: 'object',
          properties: {
            pageSize: { type: 'number', description: 'Albums to return (1–50, default 20)' },
            pageToken: { type: 'string', description: 'Pagination token from a previous call' },
          },
        },
      },
      {
        name: 'get_recent_photos',
        description: 'Get the most recent photos/videos from Google Photos in reverse chronological order.',
        parameters: {
          type: 'object',
          properties: {
            pageSize: { type: 'number', description: 'Photos to return (1–100, default 25)' },
            pageToken: { type: 'string' },
          },
        },
      },
      {
        name: 'get_album_photos',
        description: 'List photos/videos in a specific Google Photos album by albumId.',
        parameters: {
          type: 'object',
          properties: {
            albumId: { type: 'string', description: 'Album ID from list_albums' },
            pageSize: { type: 'number', description: 'Items to return (1–100, default 25)' },
            pageToken: { type: 'string' },
          },
          required: ['albumId'],
        },
      },
      {
        name: 'search_photos',
        description: 'Search Google Photos by date range and/or content categories (SPORT, SELFIES, FOOD, PEOPLE, TRAVEL, etc.).',
        parameters: {
          type: 'object',
          properties: {
            startDate: { type: 'string', description: 'Start date YYYY-MM-DD' },
            endDate: { type: 'string', description: 'End date YYYY-MM-DD' },
            contentCategories: {
              type: 'array',
              items: { type: 'string' },
              description: 'ANIMALS, BIRTHDAYS, CITYSCAPES, DOCUMENTS, FOOD, LANDMARKS, LANDSCAPES, NIGHT, PEOPLE, PERFORMANCES, PETS, SCREENSHOTS, SELFIES, SHOPPING, SPORT, TRAVEL, WEDDINGS',
            },
            mediaType: { type: 'string', description: 'ALL_MEDIA, VIDEO, or PHOTO' },
            pageSize: { type: 'number', description: '1–100, default 25' },
            pageToken: { type: 'string' },
          },
        },
      },
    ],
  },
];

// -----------------------------------------------------------------------

class GeminiProvider implements AgentProvider {
  readonly supportsNativeSlashCommands = false;
  private ai: GoogleGenAI;
  private modelName: string;

  constructor(_opts: ProviderOptions) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) throw new Error('GEMINI_API_KEY is not set — add it to .env');
    this.ai = new GoogleGenAI({ apiKey });
    this.modelName = process.env.GEMINI_MODEL || 'gemini-2.0-flash';
    log(`Initialized with model ${this.modelName}`);
  }

  isSessionInvalid(_err: unknown): boolean {
    return false;
  }

  query(input: QueryInput): AgentQuery {
    let aborted = false;
    const self = this;

    async function* run(): AsyncGenerator<ProviderEvent> {
      yield { type: 'init', continuation: 'gemini' };
      try {
        const history = loadHistory();
        const userParts: GeminiPart[] = [{ text: input.prompt }];
        if (input.attachments && input.attachments.length > 0) {
          userParts.push(...buildGeminiImageParts(input.attachments));
        }
        history.push({ role: 'user', parts: userParts });

        const systemInstruction = input.systemContext?.instructions;
        let contents = [...history];
        let finalText = '';

        for (let round = 0; round < MAX_TOOL_ROUNDS && !aborted; round++) {
          const stream = self.ai.models.generateContentStream({
            model: self.modelName,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            config: { tools: TOOLS as any, ...(systemInstruction ? { systemInstruction } : {}) },
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            contents: contents as any,
          });

          let chunkText = '';
          const functionCalls: Array<{ name: string; args: Record<string, unknown> }> = [];

          for await (const chunk of await stream) {
            if (aborted) break;
            yield { type: 'activity' };

            for (const part of chunk.candidates?.[0]?.content?.parts ?? []) {
              if ('text' in part && part.text) chunkText += part.text;
              if ('functionCall' in part && part.functionCall?.name) {
                functionCalls.push({
                  name: part.functionCall.name,
                  args: (part.functionCall.args ?? {}) as Record<string, unknown>,
                });
              }
            }
          }

          if (aborted) break;

          // Record model turn
          const modelParts: GeminiPart[] = [];
          if (chunkText) modelParts.push({ text: chunkText });
          for (const fc of functionCalls) modelParts.push({ functionCall: { name: fc.name, args: fc.args } });
          contents.push({ role: 'model', parts: modelParts });

          if (functionCalls.length === 0) {
            finalText = chunkText;
            break;
          }

          // Execute tools and feed results back
          const toolParts: GeminiPart[] = [];
          for (const fc of functionCalls) {
            yield { type: 'progress', message: `→ ${fc.name}` };
            const output = await executeTool(fc.name, fc.args);
            toolParts.push({ functionResponse: { name: fc.name, response: { output } } });
          }
          contents.push({ role: 'user', parts: toolParts });
        }

        saveHistory(contents);
        yield { type: 'result', text: finalText || null };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        log(`Error: ${message}`);
        yield { type: 'error', message, retryable: false };
      }
    }

    return {
      push(_msg: string) { /* single-turn for now */ },
      end() {},
      abort() { aborted = true; },
      events: run(),
    };
  }
}

registerProvider('gemini', (opts) => new GeminiProvider(opts));
