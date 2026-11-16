// 通信がなくても起動できるように、アプリの部品をスマホの中に保存しておく仕組み（サービスワーカー）。
// 撮影した映像はここでは一切扱わない。保存するのはアプリの部品だけ。

const VERSION = 'v1';
const CACHE = 'kaigo-camera-' + VERSION;

// 最初に開いたときに、まとめて保存する部品
const PRECACHE = [
  './',
  './index.html',
  './style.css',
  './app.js',
  './camera.js',
  './recorder.js',
  './face-id.js',
  './manifest.json',
  './icon-180.png',
  './icon-512.png',
  './vendor/tasks-vision/vision_bundle.mjs',
  './vendor/tasks-vision/wasm/vision_wasm_internal.js',
  './vendor/tasks-vision/wasm/vision_wasm_internal.wasm',
  './vendor/blaze_face_short_range.tflite',
  './vendor/face-api/face-api.esm.js',
  './vendor/face-api/model/face_recognition_model-weights_manifest.json',
  './vendor/face-api/model/face_recognition_model.bin',
];

// 保存してよい種類（動画などは保存しない）
const KEEP = /\.(html|css|js|mjs|json|png|wasm|tflite|bin)$/i;

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(PRECACHE)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith('kaigo-camera-') && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  const isPage = req.mode === 'navigate';
  if (!isPage && !KEEP.test(url.pathname)) return;

  if (url.pathname.includes('/vendor/')) {
    // 顔検出などの大きな部品は中身が変わらないので、保存してあるものを優先（速く、通信量も少ない）
    event.respondWith(
      caches.match(req, { ignoreSearch: true }).then((hit) => hit || fetchAndKeep(req))
    );
    return;
  }
  // アプリ本体は、通信できるときは新しいものを取りに行き、できないときは保存してあるものを使う
  event.respondWith(
    fetchAndKeep(req).catch(() =>
      caches.match(req, { ignoreSearch: true }).then((hit) => hit || (isPage ? caches.match('./index.html') : Response.error()))
    )
  );
});

async function fetchAndKeep(req) {
  const res = await fetch(req);
  if (res && res.ok && res.type === 'basic') {
    const copy = res.clone();
    caches.open(CACHE).then((cache) => cache.put(req, copy)).catch(() => {});
  }
  return res;
}
