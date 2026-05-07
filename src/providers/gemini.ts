/**
 * Host-side container config for the `gemini` provider.
 * Passes GEMINI_API_KEY and GEMINI_MODEL from the host .env into the container.
 *
 * readEnvFile does NOT set process.env (by design — keeps secrets out of child
 * processes). So we read directly from .env here rather than ctx.hostEnv.
 */
import { readEnvFile } from '../env.js';
import { registerProviderContainerConfig } from './provider-container-registry.js';

function mergeNoProxy(current: string | undefined, addition: string): string {
  if (!current?.trim()) return addition;
  const parts = new Set(current.split(/[\s,]+/).map((s) => s.trim()).filter(Boolean));
  parts.add(addition);
  return [...parts].join(',');
}

registerProviderContainerConfig('gemini', (_ctx) => {
  const file = readEnvFile(['GEMINI_API_KEY', 'GEMINI_MODEL']);
  const env: Record<string, string> = {};
  if (file.GEMINI_API_KEY) env.GEMINI_API_KEY = file.GEMINI_API_KEY;
  if (file.GEMINI_MODEL) env.GEMINI_MODEL = file.GEMINI_MODEL;
  // Bypass OneCLI proxy for Google's API — the SDK sends x-goog-api-key from
  // the env var above; OneCLI would inject Authorization: Bearer which Google
  // rejects as ACCESS_TOKEN_TYPE_UNSUPPORTED.
  const bypass = 'generativelanguage.googleapis.com';
  env.NO_PROXY = mergeNoProxy(process.env.NO_PROXY, bypass);
  env.no_proxy = mergeNoProxy(process.env.no_proxy, bypass);
  return { env };
});
