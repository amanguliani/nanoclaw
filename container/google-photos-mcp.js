#!/usr/bin/env node
'use strict';

// Google Photos Library API — minimal MCP server (stdio, newline-delimited JSON-RPC 2.0)
// OneCLI intercepts outbound calls to photoslibrary.googleapis.com and swaps the bearer token.
// Stub credentials: GOOGLE_PHOTOS_TOKEN_PATH (default ~/.google-photos-mcp/token.json)

const https = require('https');
const http = require('http');
const fs = require('fs');
const readline = require('readline');

const TOKEN_PATH =
  process.env.GOOGLE_PHOTOS_TOKEN_PATH ||
  `${process.env.HOME || '/root'}/.google-photos-mcp/token.json`;

function getToken() {
  try {
    return JSON.parse(fs.readFileSync(TOKEN_PATH, 'utf8')).access_token || 'onecli-managed';
  } catch {
    return 'onecli-managed';
  }
}

function photosRequest(path, method, body) {
  return new Promise((resolve, reject) => {
    const data = body ? JSON.stringify(body) : null;
    const isProxy = !!(process.env.HTTPS_PROXY || process.env.https_proxy);
    const proxyUrl = process.env.HTTPS_PROXY || process.env.https_proxy || '';

    const headers = {
      Authorization: `Bearer ${getToken()}`,
      'Content-Type': 'application/json',
      ...(data ? { 'Content-Length': String(Buffer.byteLength(data)) } : {}),
    };

    let options;
    let transport;

    if (isProxy) {
      const proxy = new URL(proxyUrl);
      options = {
        hostname: proxy.hostname,
        port: proxy.port || 8080,
        path: `https://photoslibrary.googleapis.com${path}`,
        method: method || 'GET',
        headers: { ...headers, Host: 'photoslibrary.googleapis.com' },
      };
      transport = http;
    } else {
      options = {
        hostname: 'photoslibrary.googleapis.com',
        path,
        method: method || 'GET',
        headers,
      };
      transport = https;
    }

    const req = transport.request(options, (res) => {
      let buf = '';
      res.on('data', (c) => { buf += c; });
      res.on('end', () => {
        try { resolve(JSON.parse(buf)); } catch { resolve({ _raw: buf }); }
      });
    });
    req.on('error', reject);
    if (data) req.write(data);
    req.end();
  });
}

function parseDate(str) {
  if (!str) return null;
  const [y, m, d] = str.split('-').map(Number);
  return { year: y || 0, month: m || 0, day: d || 0 };
}

const TOOLS = [
  {
    name: 'list_albums',
    description: 'List all Google Photos albums. Returns album IDs, titles, and item counts.',
    inputSchema: {
      type: 'object',
      properties: {
        pageSize: { type: 'integer', description: 'Albums to return (1–50, default 20)' },
        pageToken: { type: 'string', description: 'Pagination token from a previous call' },
      },
    },
  },
  {
    name: 'get_album_photos',
    description: 'List photos/videos in a specific album.',
    inputSchema: {
      type: 'object',
      required: ['albumId'],
      properties: {
        albumId: { type: 'string', description: 'Album ID from list_albums' },
        pageSize: { type: 'integer', description: 'Items to return (1–100, default 25)' },
        pageToken: { type: 'string' },
      },
    },
  },
  {
    name: 'get_recent_photos',
    description: 'Get the most recent photos/videos in reverse chronological order.',
    inputSchema: {
      type: 'object',
      properties: {
        pageSize: { type: 'integer', description: 'Photos to return (1–100, default 25)' },
        pageToken: { type: 'string' },
      },
    },
  },
  {
    name: 'search_photos',
    description: 'Search photos by date range and/or content categories.',
    inputSchema: {
      type: 'object',
      properties: {
        startDate: { type: 'string', description: 'Start date YYYY-MM-DD' },
        endDate: { type: 'string', description: 'End date YYYY-MM-DD' },
        contentCategories: {
          type: 'array',
          items: { type: 'string' },
          description: 'ANIMALS, BIRTHDAYS, CITYSCAPES, DOCUMENTS, FASHION, FLOWERS, FOOD, GARDENS, HOLIDAYS, HOUSES, LANDMARKS, LANDSCAPES, NIGHT, PEOPLE, PERFORMANCES, PETS, RECEIPTS, SCREENSHOTS, SELFIES, SHOPPING, SPORT, TRAVEL, WEDDINGS, WHITEBOARDS',
        },
        mediaType: {
          type: 'string',
          enum: ['ALL_MEDIA', 'VIDEO', 'PHOTO'],
          description: 'Filter by media type (default ALL_MEDIA)',
        },
        pageSize: { type: 'integer', description: '1–100, default 25' },
        pageToken: { type: 'string' },
      },
    },
  },
  {
    name: 'get_photo',
    description: 'Get metadata and download URL for a specific photo/video by ID.',
    inputSchema: {
      type: 'object',
      required: ['mediaItemId'],
      properties: {
        mediaItemId: { type: 'string', description: 'Media item ID from a previous search' },
      },
    },
  },
];

async function callTool(name, args) {
  switch (name) {
    case 'list_albums': {
      const ps = Math.min(Number(args.pageSize) || 20, 50);
      const q = `pageSize=${ps}${args.pageToken ? `&pageToken=${encodeURIComponent(args.pageToken)}` : ''}`;
      return photosRequest(`/v1/albums?${q}`);
    }
    case 'get_album_photos': {
      const body = { albumId: args.albumId, pageSize: Math.min(Number(args.pageSize) || 25, 100) };
      if (args.pageToken) body.pageToken = args.pageToken;
      return photosRequest('/v1/mediaItems:search', 'POST', body);
    }
    case 'get_recent_photos': {
      const ps = Math.min(Number(args.pageSize) || 25, 100);
      const q = `pageSize=${ps}${args.pageToken ? `&pageToken=${encodeURIComponent(args.pageToken)}` : ''}`;
      return photosRequest(`/v1/mediaItems?${q}`);
    }
    case 'search_photos': {
      const filters = {};
      if (args.startDate || args.endDate) {
        filters.dateFilter = {
          ranges: [{ startDate: parseDate(args.startDate), endDate: parseDate(args.endDate) }],
        };
      }
      if (Array.isArray(args.contentCategories) && args.contentCategories.length > 0) {
        filters.contentFilter = { includedContentCategories: args.contentCategories };
      }
      if (args.mediaType && args.mediaType !== 'ALL_MEDIA') {
        filters.mediaTypeFilter = { mediaTypes: [args.mediaType] };
      }
      const body = { pageSize: Math.min(Number(args.pageSize) || 25, 100), filters };
      if (args.pageToken) body.pageToken = args.pageToken;
      return photosRequest('/v1/mediaItems:search', 'POST', body);
    }
    case 'get_photo':
      return photosRequest(`/v1/mediaItems/${encodeURIComponent(String(args.mediaItemId))}`);
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

// ─── MCP stdio server (newline-delimited JSON-RPC 2.0) ────────────────────────

const rl = readline.createInterface({ input: process.stdin, terminal: false });

rl.on('line', async (line) => {
  line = line.trim();
  if (!line) return;
  let req;
  try { req = JSON.parse(line); } catch { return; }

  const id = req.id ?? null;
  let response;

  try {
    switch (req.method) {
      case 'initialize':
        response = {
          jsonrpc: '2.0', id,
          result: {
            protocolVersion: '2024-11-05',
            capabilities: { tools: {} },
            serverInfo: { name: 'google-photos-mcp', version: '1.0.0' },
          },
        };
        break;
      case 'tools/list':
        response = { jsonrpc: '2.0', id, result: { tools: TOOLS } };
        break;
      case 'tools/call': {
        const result = await callTool(req.params?.name, req.params?.arguments || {});
        response = {
          jsonrpc: '2.0', id,
          result: { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] },
        };
        break;
      }
      case 'notifications/initialized':
      case 'notifications/cancelled':
        return;
      default:
        response = { jsonrpc: '2.0', id, error: { code: -32601, message: 'Method not found' } };
    }
  } catch (err) {
    response = { jsonrpc: '2.0', id, error: { code: -32603, message: String(err?.message || err) } };
  }

  if (response) process.stdout.write(JSON.stringify(response) + '\n');
});

process.stdin.resume();
