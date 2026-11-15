// カメラ取得・顔検出・ぼかし描画・タップ切り替え・追跡
//
// 安全の考え方：
//  - カメラ映像は「raw」という内部の画面にだけ描く（録画されない）
//  - 録画されるのは「out」だけ。outには必ず、ぼかしをかけた後の絵しか描かない
//  - 顔検出の準備ができていない間は、画面全体をぼかす
//  - 顔を見失っても、しばらくは直前の位置をぼかし続ける

import { FilesetResolver, FaceDetector } from './vendor/tasks-vision/vision_bundle.mjs';

const HOLD_BLUR_MS = 1200;   // ぼかす顔を見失ってから、ぼかしを残す時間
const HOLD_CLEAR_MS = 800;   // ぼかさない顔を見失ってから、選択を覚えておく時間（過ぎると選択解除＝ぼかしに戻る）
const EXPAND_W = 1.7;        // ぼかし範囲の広げ方（顔の幅に対して）
const EXPAND_H = 2.0;
const BLUR_CELLS = 7;        // 小さいほど強いぼかし

export class BlurCamera {
  constructor({ video, out, overlay, onStatus }) {
    this.video = video;
    this.out = out;
    this.octx = out.getContext('2d', { alpha: false });
    this.overlay = overlay;
    this.vctx = overlay.getContext('2d');
    this.raw = document.createElement('canvas');
    this.rctx = this.raw.getContext('2d', { alpha: false, willReadFrequently: false });
    this.tile = document.createElement('canvas');
    this.tctx = this.tile.getContext('2d', { alpha: false });
    this.small = document.createElement('canvas');
    this.sctx = this.small.getContext('2d', { alpha: false });
    this.onStatus = onStatus || (() => {});

    this.detector = null;
    this.detectorFailed = false;
    this.tracks = [];
    this.nextId = 1;
    this.facing = 'environment';
    this.stream = null;
    this.running = false;
    this.locked = false;      // 録画中は画面サイズを固定
    this.label = '';
    this.tileIndex = 0;
    this.lastTs = 0;
    this._loop = this._loop.bind(this);

    out.addEventListener('pointerdown', (e) => this._onTap(e));
  }

  get ready() { return !!this.detector && this.running && this.out.width > 0; }

  async initDetector() {
    if (this.detector) return;
    const vision = await FilesetResolver.forVisionTasks('./vendor/tasks-vision/wasm');
    const make = (delegate) => FaceDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: './vendor/blaze_face_short_range.tflite', delegate },
      runningMode: 'VIDEO',
      minDetectionConfidence: 0.3,
      minSuppressionThreshold: 0.3,
    });
    try {
      this.detector = await make('GPU');
    } catch (e) {
      this.detector = await make('CPU');
    }
  }

  async start() {
    await this._openCamera();
    this.running = true;
    requestAnimationFrame(this._loop);
    try {
      await this.initDetector();
    } catch (e) {
      this.detectorFailed = true;
      throw e;
    }
  }

  async _openCamera() {
    this._closeCamera();
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: this.facing }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    this.tracks = [];
  }

  _closeCamera() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  async flip() {
    if (this.locked) return;
    this.facing = this.facing === 'environment' ? 'user' : 'environment';
    await this._openCamera();
  }

  stop() {
    this.running = false;
    this._closeCamera();
    this.tracks = [];
  }

  lock(v) { this.locked = v; }
  setLabel(text) { this.label = text; }

  // ---- 毎フレームの処理 ----
  _loop() {
    if (!this.running) return;
    requestAnimationFrame(this._loop);
    const v = this.video;
    const vw = v.videoWidth, vh = v.videoHeight;
    if (!vw || !vh || v.readyState < 2) return;

    // 録画中でなければ、カメラの向きに合わせて画面サイズを決める
    if (!this.locked && (this.out.width !== vw || this.out.height !== vh)) {
      for (const c of [this.out, this.raw, this.overlay]) { c.width = vw; c.height = vh; }
      this.tracks = [];
    }
    const W = this.out.width, H = this.out.height;

    // 1) カメラ映像を内部画面へ（録画中に向きが変わったら、枠内に収めて描く）
    if (vw === W && vh === H) {
      this.rctx.drawImage(v, 0, 0, W, H);
    } else {
      const s = Math.min(W / vw, H / vh);
      const dw = vw * s, dh = vh * s;
      this.rctx.fillStyle = '#000';
      this.rctx.fillRect(0, 0, W, H);
      this.rctx.drawImage(v, (W - dw) / 2, (H - dh) / 2, dw, dh);
    }

    const now = performance.now();

    // 2) 顔検出
    let detected = false;
    if (this.detector) {
      try {
        this._updateTracks(this._detect(W, H, now), now);
        detected = true;
      } catch (e) {
        detected = false;
      }
    }

    // 3) 録画される画面へ描く
    if (!detected) {
      // 検出できない間は全体をぼかす
      this._blurAll(W, H);
    } else {
      this.octx.drawImage(this.raw, 0, 0);
      for (const t of this.tracks) {
        if (!t.clear) this._blurFace(t, W, H);
      }
    }
    this._drawLabel(W, H);

    // 4) 目印（録画されない）
    this._drawOverlay(W, H);
  }

  _ts(now) {
    const ts = Math.max(now, this.lastTs + 1);
    this.lastTs = ts;
    return ts;
  }

  _detect(W, H, now) {
    const boxes = [];
    const full = this.detector.detectForVideo(this.raw, this._ts(now));
    for (const d of full.detections) {
      const b = d.boundingBox;
      if (b) boxes.push({ x: b.originX, y: b.originY, w: b.width, h: b.height });
    }
    // 遠くの小さい顔を拾うため、画面の一部を拡大して順番に調べる
    const tw = Math.round(W * 0.6), th = Math.round(H * 0.6);
    const pos = [[0, 0], [W - tw, 0], [0, H - th], [W - tw, H - th], [(W - tw) / 2, (H - th) / 2]];
    const [tx, ty] = pos[this.tileIndex];
    this.tileIndex = (this.tileIndex + 1) % pos.length;
    if (this.tile.width !== tw || this.tile.height !== th) { this.tile.width = tw; this.tile.height = th; }
    this.tctx.drawImage(this.raw, tx, ty, tw, th, 0, 0, tw, th);
    const part = this.detector.detectForVideo(this.tile, this._ts(now));
    for (const d of part.detections) {
      const b = d.boundingBox;
      if (b) boxes.push({ x: b.originX + tx, y: b.originY + ty, w: b.width, h: b.height });
    }
    return boxes;
  }

  _updateTracks(boxes, now) {
    for (const b of boxes) {
      let best = null, bestScore = 0;
      for (const t of this.tracks) {
        const s = overlapScore(t, b);
        if (s > bestScore) { bestScore = s; best = t; }
      }
      if (best && bestScore > 0.15) {
        const a = 0.65;
        best.x += (b.x - best.x) * a; best.y += (b.y - best.y) * a;
        best.w += (b.w - best.w) * a; best.h += (b.h - best.h) * a;
        best.seen = now;
      } else {
        this.tracks.push({ id: this.nextId++, x: b.x, y: b.y, w: b.w, h: b.h, seen: now, clear: false });
      }
    }
    // 見失った顔の整理。ぼかさない顔は、見失ったら選択を解除する（次に現れたらぼかす）
    this.tracks = this.tracks.filter((t) => now - t.seen < (t.clear ? HOLD_CLEAR_MS : HOLD_BLUR_MS));
  }

  _region(t, W, H) {
    const cx = t.x + t.w / 2, cy = t.y + t.h / 2 - t.h * 0.08;
    const w = t.w * EXPAND_W, h = t.h * EXPAND_H;
    let x = cx - w / 2, y = cy - h / 2;
    const x0 = Math.max(0, Math.floor(x)), y0 = Math.max(0, Math.floor(y));
    const x1 = Math.min(W, Math.ceil(x + w)), y1 = Math.min(H, Math.ceil(y + h));
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  }

  _blurFace(t, W, H) {
    const r = this._region(t, W, H);
    if (r.w < 2 || r.h < 2) return;
    const sw = BLUR_CELLS, sh = Math.max(2, Math.round(BLUR_CELLS * r.h / r.w));
    if (this.small.width !== sw || this.small.height !== sh) { this.small.width = sw; this.small.height = sh; }
    this.sctx.imageSmoothingEnabled = true;
    this.sctx.drawImage(this.raw, r.x, r.y, r.w, r.h, 0, 0, sw, sh);
    const o = this.octx;
    o.save();
    o.beginPath();
    o.ellipse(r.x + r.w / 2, r.y + r.h / 2, r.w / 2, r.h / 2, 0, 0, Math.PI * 2);
    o.clip();
    o.imageSmoothingEnabled = true;
    o.drawImage(this.small, 0, 0, sw, sh, r.x, r.y, r.w, r.h);
    o.restore();
  }

  _blurAll(W, H) {
    const sw = 16, sh = Math.max(2, Math.round(16 * H / W));
    if (this.small.width !== sw || this.small.height !== sh) { this.small.width = sw; this.small.height = sh; }
    this.sctx.drawImage(this.raw, 0, 0, W, H, 0, 0, sw, sh);
    this.octx.imageSmoothingEnabled = true;
    this.octx.drawImage(this.small, 0, 0, sw, sh, 0, 0, W, H);
  }

  _drawLabel(W, H) {
    if (!this.label) return;
    const o = this.octx;
    const fs = Math.round(Math.min(W, H) * 0.042);
    o.save();
    o.font = `700 ${fs}px -apple-system, "Hiragino Sans", "Yu Gothic UI", Meiryo, sans-serif`;
    o.textBaseline = 'middle';
    const pad = fs * 0.5;
    const tw = o.measureText(this.label).width;
    const x = fs * 0.6, y = H - fs * 0.6 - fs * 1.6;
    o.fillStyle = 'rgba(0,0,0,0.6)';
    o.fillRect(x, y, tw + pad * 2, fs * 1.6);
    o.fillStyle = '#fff';
    o.fillText(this.label, x + pad, y + fs * 0.82);
    o.restore();
  }

  _drawOverlay(W, H) {
    const c = this.vctx;
    c.clearRect(0, 0, W, H);
    const lw = Math.max(3, Math.round(Math.min(W, H) * 0.006));
    for (const t of this.tracks) {
      const r = this._region(t, W, H);
      c.lineWidth = lw;
      if (t.clear) {
        c.strokeStyle = '#34d399';
        c.setLineDash([]);
      } else {
        c.strokeStyle = 'rgba(255,255,255,0.75)';
        c.setLineDash([lw * 3, lw * 2]);
      }
      c.beginPath();
      c.ellipse(r.x + r.w / 2, r.y + r.h / 2, r.w / 2, r.h / 2, 0, 0, Math.PI * 2);
      c.stroke();
      if (t.clear) {
        const fs = Math.round(Math.min(W, H) * 0.032);
        c.font = `700 ${fs}px sans-serif`;
        c.textAlign = 'center';
        c.fillStyle = '#34d399';
        c.fillText('ぼかしなし', r.x + r.w / 2, Math.max(fs, r.y - fs * 0.4));
      }
    }
    c.setLineDash([]);
  }

  // ---- タップで切り替え ----
  _onTap(e) {
    const W = this.out.width, H = this.out.height;
    if (!W || !H) return;
    const rect = this.out.getBoundingClientRect();
    const s = Math.min(rect.width / W, rect.height / H);
    const offX = (rect.width - W * s) / 2, offY = (rect.height - H * s) / 2;
    const px = (e.clientX - rect.left - offX) / s;
    const py = (e.clientY - rect.top - offY) / s;
    let hit = null, hitArea = Infinity;
    for (const t of this.tracks) {
      const r = this._region(t, W, H);
      if (px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h && r.w * r.h < hitArea) {
        hit = t; hitArea = r.w * r.h;
      }
    }
    if (hit) hit.clear = !hit.clear;
  }
}

function overlapScore(t, b) {
  const x0 = Math.max(t.x, b.x), y0 = Math.max(t.y, b.y);
  const x1 = Math.min(t.x + t.w, b.x + b.w), y1 = Math.min(t.y + t.h, b.y + b.h);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const iou = inter / (t.w * t.h + b.w * b.h - inter || 1);
  // 速く動いて重なりが小さくても、中心が近ければ同じ顔とみなす
  const dx = (t.x + t.w / 2) - (b.x + b.w / 2), dy = (t.y + t.h / 2) - (b.y + b.h / 2);
  const dist = Math.hypot(dx, dy) / Math.max(t.w, b.w, 1);
  const near = dist < 0.7 ? (0.7 - dist) / 0.7 * 0.5 : 0;
  return Math.max(iou, near);
}
