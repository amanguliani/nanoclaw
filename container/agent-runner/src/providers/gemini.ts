import fs from 'fs';
import path from 'path';

import { GoogleGenAI } from '@google/genai';

import { registerProvider } from './provider-registry.js';
import type { AgentProvider, AgentQuery, ProviderEvent, ProviderOptions, QueryInput } from './types.js';

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
  | { functionCall: { name: string; args: unknown } }
  | { functionResponse: { name: string; response: { output: string } } };

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
    // Keep last 80 turns to bound token growth
    const trimmed = history.slice(-80);
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

function executeTool(name: string, args: Record<string, unknown>): string {
  switch (name) {
    case 'read_file':   return toolReadFile(args);
    case 'write_file':  return toolWriteFile(args);
    case 'list_files':  return toolListFiles(args);
    default:            return `Unknown tool: ${name}`;
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
        history.push({ role: 'user', parts: [{ text: input.prompt }] });

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
            const output = executeTool(fc.name, fc.args);
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
