// 確認用の簡易サーバー。
//   node tools/dev-server.mjs            → http://localhost:8443 （パソコンでの確認用）
//   node tools/dev-server.mjs --https    → https://<パソコンのIP>:8443 （同じWi-FiのiPhoneでの確認用。tools/cert に証明書が必要）
import http from 'node:http';
import https from 'node:https';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const port = 8443;
const types = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.wasm': 'application/wasm',
  '.png': 'image/png', '.mp4': 'video/mp4', '.mov': 'video/mp4', '.tflite': 'application/octet-stream',
};

function handler(req, res) {
  let p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  if (p.endsWith('/')) p += 'index.html';
  const file = path.join(root, p);
  if (!file.startsWith(root) || file.startsWith(path.join(root, 'tools'))) { res.writeHead(403).end(); return; }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404).end('not found'); return; }
    res.writeHead(200, { 'Content-Type': types[path.extname(file).toLowerCase()] || 'application/octet-stream', 'Cache-Control': 'no-store' });
    res.end(data);
  });
}

if (process.argv.includes('--https')) {
  const dir = path.join(root, 'tools', 'cert');
  const server = https.createServer({ key: fs.readFileSync(path.join(dir, 'key.pem')), cert: fs.readFileSync(path.join(dir, 'cert.pem')) }, handler);
  server.listen(port, '0.0.0.0', () => {
    for (const list of Object.values(os.networkInterfaces())) for (const n of list) {
      if (n.family === 'IPv4' && !n.internal) console.log(`https://${n.address}:${port}/`);
    }
  });
} else {
  http.createServer(handler).listen(port, '127.0.0.1', () => console.log(`http://localhost:${port}/`));
}
