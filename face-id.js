// 顔の特徴を数値にして、同じ人かどうかを見分ける。
// すべて端末の中で計算し、数値はメモリの中だけに持つ（保存も送信もしない）。

let faceapi = null;
let ready = false;

export function faceIdReady() { return ready; }

export async function loadFaceId() {
  if (ready) return;
  faceapi = await import('./vendor/face-api/face-api.esm.js');
  try { await faceapi.tf.setBackend('webgl'); } catch (e) { /* 使える方式に任せる */ }
  await faceapi.tf.ready();
  await faceapi.nets.faceRecognitionNet.loadFromUri('./vendor/face-api/model');
  // 最初の1回は時間がかかるので、空の画像で慣らしておく
  const warm = document.createElement('canvas');
  warm.width = 150; warm.height = 150;
  await faceapi.nets.faceRecognitionNet.computeFaceDescriptor(warm);
  ready = true;
}

// 目と口の位置を使って、顔をまっすぐ・同じ大きさに切り出す。
// kp: MediaPipeの顔の点（画面上の座標）[右目, 左目, 鼻, 口, 右耳, 左耳]
// 横顔や小さすぎる顔は見分けに使えないので false を返す。
export function cropAligned(src, kp, canvas) {
  if (!kp || kp.length < 4) return false;
  const a = kp[0], b = kp[1], m = kp[3];
  const eyeDist = Math.hypot(b.x - a.x, b.y - a.y);
  const toMouth = (Math.hypot(m.x - a.x, m.y - a.y) + Math.hypot(m.x - b.x, m.y - b.y)) / 2;
  if (toMouth < 14) return false;                 // 小さすぎる
  if (eyeDist / toMouth < 0.6) return false;      // 横を向きすぎている
  const size = toMouth / 0.45;
  const cx = (a.x + b.x + m.x) / 3, cy = (a.y + b.y + m.y) / 3;
  const angle = Math.atan2(b.y - a.y, b.x - a.x);
  const S = 150;
  if (canvas.width !== S || canvas.height !== S) { canvas.width = S; canvas.height = S; }
  const g = canvas.getContext('2d');
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.fillStyle = '#000';
  g.fillRect(0, 0, S, S);
  g.translate(S * 0.5, S * 0.43);
  g.rotate(-angle);
  g.scale(S / size, S / size);
  g.translate(-cx, -cy);
  g.imageSmoothingEnabled = true;
  g.drawImage(src, 0, 0);
  g.setTransform(1, 0, 0, 1, 0, 0);
  return true;
}

export async function describe(canvas) {
  return faceapi.nets.faceRecognitionNet.computeFaceDescriptor(canvas);
}

export function distance(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}
