// ぼかし済み画面（canvas）だけを録画する。カメラの生映像は録画の経路に入れない。

const TYPES = [
  'video/mp4;codecs=avc1.42E01E,mp4a.40.2',
  'video/mp4',
  'video/webm;codecs=vp9,opus',
  'video/webm;codecs=vp8,opus',
  'video/webm',
];

export function pickMimeType() {
  if (typeof MediaRecorder === 'undefined') return null;
  for (const t of TYPES) {
    try { if (MediaRecorder.isTypeSupported(t)) return t; } catch (e) { /* 次へ */ }
  }
  return '';
}

export class CanvasRecorder {
  constructor(canvas) {
    this.canvas = canvas;
    this.rec = null;
    this.chunks = [];
    this.audioStream = null;
    this.startedAt = 0;
  }

  get recording() { return !!this.rec && this.rec.state === 'recording'; }
  get elapsedMs() { return this.recording ? performance.now() - this.startedAt : 0; }

  async start({ audio }) {
    const mime = pickMimeType();
    if (mime === null) throw new Error('この端末のブラウザは録画に対応していません');
    if (!this.canvas.captureStream) throw new Error('この端末のブラウザは画面の録画に対応していません');

    const stream = this.canvas.captureStream(30);
    let audioOk = false;
    if (audio) {
      try {
        this.audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        for (const t of this.audioStream.getAudioTracks()) stream.addTrack(t);
        audioOk = true;
      } catch (e) {
        audioOk = false; // マイクが使えなければ音声なしで続ける
      }
    }
    const opts = { videoBitsPerSecond: 5_000_000 };
    if (mime) opts.mimeType = mime;
    this.chunks = [];
    this.rec = new MediaRecorder(stream, opts);
    this.rec.ondataavailable = (e) => { if (e.data && e.data.size) this.chunks.push(e.data); };
    this.rec.start(1000);
    this.startedAt = performance.now();
    return { audioOk };
  }

  stop() {
    return new Promise((resolve, reject) => {
      const rec = this.rec;
      if (!rec) return reject(new Error('録画していません'));
      const durationMs = performance.now() - this.startedAt;
      rec.onerror = (e) => reject(e.error || new Error('録画に失敗しました'));
      rec.onstop = () => {
        const type = (rec.mimeType || this.chunks[0]?.type || 'video/mp4').split(';')[0];
        const blob = new Blob(this.chunks, { type });
        this.chunks = [];
        this._release();
        resolve({ blob, type, durationMs });
      };
      rec.stop();
    });
  }

  _release() {
    if (this.audioStream) this.audioStream.getTracks().forEach((t) => t.stop());
    this.audioStream = null;
    this.rec = null;
  }
}

export function buildFileName(title, date, type) {
  const p = (n) => String(n).padStart(2, '0');
  const d = `${date.getFullYear()}-${p(date.getMonth() + 1)}-${p(date.getDate())}`;
  const t = `${p(date.getHours())}${p(date.getMinutes())}`;
  const ext = type.includes('webm') ? 'webm' : 'mp4';
  const safe = title.replace(/[\\/:*?"<>|\s]+/g, '_');
  return `${safe}_${d}_${t}.${ext}`;
}

// 共有メニュー（写真に保存・ファイルに保存・AirDrop）を開く。使えない端末ではダウンロードにする
export async function saveVideo(blob, fileName) {
  const file = new File([blob], fileName, { type: blob.type });
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: fileName });
      return 'shared';
    } catch (e) {
      if (e && e.name === 'AbortError') return 'cancelled';
      // 共有に失敗したらダウンロードへ
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = fileName;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
  return 'downloaded';
}
