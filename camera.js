// カメラ取得・顔検出・ぼかし描画・タップ切り替え・追跡
//
// 安全の考え方：
//  - カメラ映像は「raw」という内部の画面にだけ描く（録画されない）
//  - 録画されるのは「out」だけ。outには必ず、ぼかしをかけた後の絵しか描かない
//  - 顔検出の準備ができていない間は、画面全体をぼかす
//  - 顔を見失っても、しばらくは直前の位置をぼかし続ける

import { FilesetResolver, FaceDetector } from './vendor/tasks-vision/vision_bundle.mjs';
import { loadFaceId, faceIdReady, cropAligned, describe, distance } from './face-id.js';

const HOLD_BLUR_MS = 1200;   // ぼかす顔を見失ってから、ぼかしを残す時間
const HOLD_CLEAR_MS = 800;   // ぼかさない顔を見失ってから、選択を覚えておく時間（過ぎると選択解除＝ぼかしに戻る）
const HOLD_EDGE_MS = 3500;   // 画面の端で見失った顔（半分はみ出した顔は検出しにくい）は、長めにぼかしを残す
const EXPAND_W = 1.9;        // ぼかし範囲の広げ方（顔の幅に対して）。ふちをやわらかくする分、広めにとる
const EXPAND_H = 2.2;
// 顔かどうかの調べ直し（ライトや家具を顔とまちがえた枠を消すため）
const HOLD_UNSURE_MS = 550;  // まだ顔と確かめられていない枠を見失ったとき、ぼかしを残す時間
// 判断は1回の点数ではなく、くり返し調べた平均で行う（ライトは 0.51〜0.68 を行き来し、本物の顔は 0.75〜0.94 で安定）
const VERIFY_OK = 0.72;      // 平均がこれ以上なら「顔」
const VERIFY_NG = 0.67;      // 3回以上調べて平均がこれ未満なら「顔ではない」として、ぼかしを消す
const VERIFY_STRONG = 0.8;   // 平均が一度でもこれ以上になった枠は、その後どう見えても「顔ではない」にはしない（横を向いた顔を守る）
const VERIFY_MIN_SAMPLES = 3;
// 回転させた向き（横倒し・逆さ）では基準を上げる。実測：寝ている顔は 0.75〜0.89（中央0.82）、
// ライトは回転させると 0.72 を超えることがあるが 0.8 には届かない
const VERIFY_OK_TURNED = 0.76;
const VERIFY_NG_TURNED = 0.72;
const VERIFY_UNSURE_MAX = 6;  // 判断がつかない状態がこの回数続いたら「顔ではない」とする（中間の点数をうろうろするライト対策）
const VERIFY_EVERY_MS = 150; // 判断がつくまでの調べ直しの間隔
const VERIFY_KEEP_MS = 500;  // 判断がついたあとも調べ続ける間隔
const STRONG_SCORE = 0.75;   // 「顔ではない」とした枠でも、検出の点数がこれ以上になったらすぐぼかしに戻す
const FAKE_ZONE_MS = 8000;   // 「顔ではない」と分かった場所を覚えておく時間。同じ場所に出た枠は、ぼかす前に調べる
const BLUR_STEPS = [2, 3, 4]; // にじませる幅（顔の範囲を72に縮めた画面での幅）。大きいほど強いぼかし
const BLUR_HAZE = 0.12;       // 白っぽさ（0〜1）

// 顔の見分け（距離が小さいほど似ている。テストでは同じ人 0.33〜0.52、別人 0.69以上）
const ID_MATCH = 0.47;       // これ未満が2回続いたら「選んだ人」とみなしてぼかしを外す
const ID_MATCH_TIMES = 2;
const ID_MISMATCH = 0.68;    // ぼかしなしの顔が、これ以上の距離を3回続けたら別人とみなしてぼかしに戻す
const ID_MISMATCH_TIMES = 3;
const ID_LEARN_MIN = 0.25;   // 覚えている顔とこの範囲で違うとき、新しい角度として追加で覚える
const ID_MAX_SAMPLES = 15;
const ID_CHECK_BLUR_MS = 300;   // ぼかし中の顔を照合する間隔
const ID_CHECK_CLEAR_MS = 600;  // ぼかしなしの顔を確かめる間隔

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
    this.pre = document.createElement('canvas');
    this.pctx = this.pre.getContext('2d', { alpha: false });
    this.midA = document.createElement('canvas');
    this.actx = this.midA.getContext('2d', { alpha: false });
    this.midB = document.createElement('canvas');
    this.bctx = this.midB.getContext('2d');
    this.onToggle = null;
    this.persons = [];        // ぼかさないと選んだ人の顔の特徴（メモリの中だけ。カメラを閉じると消える）
    this.nextPerson = 1;
    this.idBusy = false;
    this.idCanvas = document.createElement('canvas');
    this.fakeZones = [];
    this.vCanvas = document.createElement('canvas');
    this.vCanvas.width = 256; this.vCanvas.height = 256;
    this.vctx2 = this.vCanvas.getContext('2d', { alpha: false });
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
    this.slotIndex = 0;
    this.searchTurned = true;   // 寝ている人など、横倒し・逆さの顔も探すか
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
    // 顔を見分ける部品も準備中に読み込む（読み込み中は映像が一瞬止まるため、撮影前に済ませる）。
    // 失敗しても、20秒かかっても、撮影はできる（その場合は位置の追跡だけになる）
    await Promise.race([
      loadFaceId().catch(() => {}),
      new Promise((r) => setTimeout(r, 20000)),
    ]);
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
    this.persons = [];
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
        this._verify(now);
        detected = true;
        this._identify(now);
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
        if (!t.clear && !t.fake) this._blurFace(t, W, H);
      }
    }
    this._drawLabel(W, H);

    // 4) 目印（録画されない）
    this._drawOverlay(W, H);
  }

  // ---- 顔かどうかの調べ直し ----
  // 疑わしい枠は、その部分だけを拡大してもう一度検出にかける。本物の顔は拡大すると高い点数で
  // 検出され直すが、ライトや家具はほとんど検出されない。1コマに1つだけ調べる。
  _verify(now) {
    let pick = null;
    for (const t of this.tracks) {
      if (t.clear || t.seen !== now || now < t.nextV) continue;
      // まだ判断がついていない枠を先に、次に「顔ではない」とした枠、最後に「顔」と分かっている枠
      const rank = (x) => (x.fake ? 1 : x.verified ? 2 : 0);
      if (!pick || rank(t) < rank(pick) || (rank(t) === rank(pick) && t.nextV < pick.nextV)) pick = t;
    }
    if (!pick) return;
    let usedRot = pick.rot || 0;
    let score = this._cropScore(pick, usedRot, now);
    if (score < VERIFY_OK && !pick.strong && this.searchTurned) {
      // まっすぐで顔と確かめられないときは、別の向き（横倒し・逆さ）でも1つずつ試す
      const alts = [0, 90, 270, 180].filter((r) => r !== (pick.rot || 0));
      const alt = alts[pick.altIdx++ % alts.length];
      const s2 = this._cropScore(pick, alt, now);
      // 別の向きは、はっきり顔と分かる点数のときだけ採用する（ライトなどが通りにくいように）
      if (s2 >= VERIFY_STRONG && s2 > score) { score = s2; usedRot = alt; pick.rot = alt; pick.view = null; pick.rotAt = now; pick.s = s2; }
    }
    // 点数の平均を更新
    pick.vAvg = pick.vN ? pick.vAvg * 0.6 + score * 0.4 : score;
    pick.vN++;
    // 「ずっと顔として扱う」保護（横を向いたり、マスクで点数が下がったりしても、ぼかしを消さない）：
    //  ・まっすぐな向きで高い点数が出たら、1回で適用する（ライトなどは、まっすぐでは高い点数にならない）
    //  ・回転させた向きでは、高い点数が続いたときだけ適用する（ライトが回転でまぐれの高得点を出すことがあるため）
    if (usedRot === 0 ? score >= VERIFY_STRONG : (pick.vN >= 4 && pick.vAvg >= VERIFY_STRONG)) pick.strong = true;
    const turned = (pick.rot || 0) !== 0;
    const okLine = turned ? VERIFY_OK_TURNED : VERIFY_OK, ngLine = turned ? VERIFY_NG_TURNED : VERIFY_NG;
    if (pick.strong || pick.vAvg >= okLine) {
      pick.verified = true; pick.fake = false; pick.unsureN = 0;
    } else {
      pick.verified = false;
      pick.unsureN = (pick.unsureN || 0) + 1;
      if ((pick.vN >= VERIFY_MIN_SAMPLES && pick.vAvg < ngLine) || pick.unsureN >= VERIFY_UNSURE_MAX) {
        if (!pick.fake) { pick.fake = true; this._rememberFake(pick, now); }
      }
      // どちらでもない間は、ぼかしたまま（「顔ではない」としていた枠は、そのまま）
    }
    // 「顔ではない」としていた枠で高い点数が出たら、人が入ってきたかもしれないので、すぐ次も調べる
    const hurry = pick.fake && score >= VERIFY_OK;
    pick.nextV = now + (hurry ? 0 : pick.verified || pick.fake ? VERIFY_KEEP_MS : VERIFY_EVERY_MS);
  }

  // 枠の部分を、指定した向きに回して拡大し、もう一度検出にかけた点数を返す
  _cropScore(t, rot, now) {
    const S = 256, g = this.vctx2;
    const side = Math.max(t.w, t.h) * 2.4;
    const cx = t.x + t.w / 2, cy = t.y + t.h / 2;
    const k = S / side;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.fillStyle = '#777';
    g.fillRect(0, 0, S, S);   // 画面の外にはみ出す部分は灰色のまま
    g.translate(S / 2, S / 2);
    g.rotate(rot * Math.PI / 180);
    g.scale(k, k);
    g.translate(-cx, -cy);
    g.drawImage(this.raw, 0, 0);
    g.setTransform(1, 0, 0, 1, 0, 0);
    let score = 0;
    for (const d of this.detector.detectForVideo(this.vCanvas, this._ts(now)).detections) {
      const b = d.boundingBox;
      if (!b) continue;
      // 切り出しの中央にある検出だけを数える（となりの人の顔を拾わないため）
      const dx = b.originX + b.width / 2 - S / 2, dy = b.originY + b.height / 2 - S / 2;
      if (Math.hypot(dx, dy) > S * 0.2 || b.width < S * 0.2) continue;
      score = Math.max(score, d.categories?.[0]?.score ?? 0);
    }
    return score;
  }

  // 「顔ではない」と分かった場所を覚える
  _rememberFake(t, now) {
    const cx = t.x + t.w / 2, cy = t.y + t.h / 2;
    this.fakeZones = this.fakeZones.filter((z) => now - z.at < FAKE_ZONE_MS && Math.hypot(z.cx - cx, z.cy - cy) > z.w * 0.5);
    this.fakeZones.push({ cx, cy, w: t.w, at: now });
  }

  _inFakeZone(b, now) {
    const cx = b.x + b.w / 2, cy = b.y + b.h / 2;
    return this.fakeZones.some((z) => now - z.at < FAKE_ZONE_MS && Math.hypot(z.cx - cx, z.cy - cy) < Math.max(z.w, b.w) * 0.8);
  }

  // ---- 顔の見分け ----
  // 1回に1つの顔だけ、映像の合間に調べる（映像の動きを止めないため）
  _identify(now) {
    if (this.idBusy || !faceIdReady()) return;
    const havePersons = this.persons.some((p) => p.samples.length > 0);
    let pick = null;
    for (const t of this.tracks) {
      if (t.fake || now - t.kpAt > 120 || now < t.nextId) continue;   // いま見えている顔だけ
      if (t.clear ? !t.person : (!havePersons || t.noAuto)) continue;
      // ぼかし中の顔（戻ってきた人かもしれない）を優先
      if (!pick || (pick.clear && !t.clear) || (pick.clear === t.clear && t.nextId < pick.nextId)) pick = t;
    }
    if (!pick) return;
    pick.nextId = now + (pick.clear ? ID_CHECK_CLEAR_MS : ID_CHECK_BLUR_MS);
    if (!cropAligned(this.raw, pick.kp, this.idCanvas)) return;   // 横顔・小さい顔は判断しない
    const wasClear = pick.clear, person = pick.person;
    this.idBusy = true;
    describe(this.idCanvas).then((desc) => {
      if (!this.tracks.includes(pick) || pick.clear !== wasClear || pick.person !== person) return;
      if (pick.clear) this._checkClear(pick, desc); else this._checkBlurred(pick, desc);
    }).catch(() => {}).finally(() => { this.idBusy = false; });
  }

  // ぼかしなしの顔：角度ちがいを追加で覚える。別人に入れ替わっていたら、ぼかしに戻す
  _checkClear(t, desc) {
    const p = t.person;
    if (!this.persons.includes(p)) return;
    if (p.samples.length === 0) { p.samples.push(desc); return; }
    const d = nearest(p, desc);
    if (d >= ID_MISMATCH) {
      if (++t.misses >= ID_MISMATCH_TIMES) { t.clear = false; t.person = null; t.misses = 0; t.hits = 0; }
      return;
    }
    t.misses = 0;
    if (d > ID_LEARN_MIN && d < ID_MATCH && p.samples.length < ID_MAX_SAMPLES) p.samples.push(desc);
  }

  // ぼかし中の顔：選んだ人と確信できるときだけ、ぼかしを外す
  _checkBlurred(t, desc) {
    let best = null, bestD = Infinity;
    for (const p of this.persons) {
      if (!p.samples.length) continue;
      if (this.tracks.some((o) => o !== t && o.clear && o.person === p)) continue;   // その人はすでに映っている
      const d = nearest(p, desc);
      if (d < bestD) { bestD = d; best = p; }
    }
    if (best && bestD < ID_MATCH) {
      if (t.candidate !== best) { t.candidate = best; t.hits = 0; }
      if (++t.hits >= ID_MATCH_TIMES) { t.clear = true; t.person = best; t.hits = 0; t.misses = 0; t.candidate = null; }
    } else {
      t.hits = 0; t.candidate = null;
    }
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
      if (b) boxes.push({ x: b.originX, y: b.originY, w: b.width, h: b.height, rot: 0, s: d.categories?.[0]?.score ?? 0, kp: (d.keypoints || []).map((k) => ({ x: k.x * W, y: k.y * H })) });
    }
    // 追加の検出を1コマに1つずつ順番に行う：
    //  ・画面の一部を拡大（遠くの小さい顔を拾う）
    //  ・映像を回転（寝ている人など、横倒し・逆さの顔を拾う。顔検出はまっすぐな顔しか得意でないため）
    //  ・その両方（遠くで寝ている人の小さな顔を拾う）
    const tw = Math.round(W * 0.6), th = Math.round(H * 0.6);
    const pos = [[0, 0], [W - tw, 0], [0, H - th], [W - tw, H - th], [(W - tw) / 2, (H - th) / 2]];
    if (!this.views || this.viewsFor !== W + 'x' + H) {
      const tiles = pos.map(([tx, ty]) => ({ rot: 0, tx, ty, tw, th, q: 1 }));
      const turned = [];
      for (const rot of [90, 270]) {
        turned.push({ rot, tx: 0, ty: 0, tw: W, th: H, q: 0.5 });
        for (const [tx, ty] of pos) turned.push({ rot, tx, ty, tw, th, q: 1 });
      }
      turned.push({ rot: 180, tx: 0, ty: 0, tw: W, th: H, q: 0.5 });
      this.views = { tiles, turned };
      this.viewsFor = W + 'x' + H;
      this.slotIndex = 0;
    this.searchTurned = true;   // 寝ている人など、横倒し・逆さの顔も探すか
    }
    const { tiles, turned } = this.views;
    // まっすぐの拡大と、回転とを1コマおきに。横倒しの顔を追っている間は、その顔を見つけた見方を優先する
    this.slotIndex++;
    const lying = this.tracks.find((t) => t.view && !t.fake && now - t.seen < 1500);
    let view;
    if (this.slotIndex % 2 === 0 || !this.searchTurned) view = tiles[(this.searchTurned ? this.slotIndex >> 1 : this.slotIndex) % tiles.length];
    else if (lying && this.slotIndex % 4 === 1) view = lying.view;
    else view = turned[(this.slotIndex >> 1) % turned.length];

    const { rot, tx, ty, q } = view;
    const vw = view.tw, vh = view.th;
    const cw = Math.round((rot === 90 || rot === 270 ? vh : vw) * q), ch = Math.round((rot === 90 || rot === 270 ? vw : vh) * q);
    if (this.tile.width !== cw || this.tile.height !== ch) { this.tile.width = cw; this.tile.height = ch; }
    const g = this.tctx;
    if (rot === 90) g.setTransform(0, q, -q, 0, vh * q, 0);          // 時計回りに90度
    else if (rot === 270) g.setTransform(0, -q, q, 0, 0, vw * q);    // 反時計回りに90度
    else if (rot === 180) g.setTransform(-q, 0, 0, -q, vw * q, vh * q);
    else g.setTransform(q, 0, 0, q, 0, 0);
    g.drawImage(this.raw, -tx, -ty);
    g.setTransform(1, 0, 0, 1, 0, 0);
    // 回転・拡大した画面での座標 (u,v) を、元の画面の座標に戻す
    const back = (u, v) => {
      let X, Y;
      if (rot === 90) { X = v / q; Y = vh - u / q; }
      else if (rot === 270) { X = vw - v / q; Y = u / q; }
      else if (rot === 180) { X = vw - u / q; Y = vh - v / q; }
      else { X = u / q; Y = v / q; }
      return { x: X + tx, y: Y + ty };
    };
    for (const d of this.detector.detectForVideo(this.tile, this._ts(now)).detections) {
      const b = d.boundingBox;
      if (!b) continue;
      const p1 = back(b.originX, b.originY), p2 = back(b.originX + b.width, b.originY + b.height);
      boxes.push({ x: Math.min(p1.x, p2.x), y: Math.min(p1.y, p2.y), w: Math.abs(p2.x - p1.x), h: Math.abs(p2.y - p1.y), rot, view: rot ? view : null, s: d.categories?.[0]?.score ?? 0, kp: (d.keypoints || []).map((k) => back(k.x * cw, k.y * ch)) });
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
        // 同じ顔でも、向きのちがう見方（回転）がたまたま反応することがある。
        // 今の向きと同じ検出か、今の向きよりはっきり点数が高い検出か、今の向きでしばらく見えていないときだけ採用する
        const sameRot = (b.rot || 0) === (best.rot || 0);
        const adopt = sameRot || b.s > best.s + 0.15 || now - best.rotAt > 700;
        best.seen = now;
        if (adopt) {
          const a = 0.65;
          best.x += (b.x - best.x) * a; best.y += (b.y - best.y) * a;
          best.w += (b.w - best.w) * a; best.h += (b.h - best.h) * a;
          best.kp = b.kp; best.kpAt = now;
          best.s = sameRot ? Math.max(b.s, best.s - 0.02) : b.s;   // 同じ向きの中では、最近の高い点数を覚えておく
          best.rot = b.rot || 0; best.view = b.view || null; best.rotAt = now;
        }
        if (best.fake) {
          // はっきり顔らしい検出が来たら、すぐぼかしに戻す。
          // 位置や大きさが変わっただけなら、ぼかしは付けずに、すぐ調べ直す（カメラが動いただけのことが多い）
          const fb = best.fakeBox;
          const moved = !fb || Math.hypot(b.x + b.w / 2 - fb.cx, b.y + b.h / 2 - fb.cy) > fb.w * 0.3 || Math.abs(b.w - fb.w) > fb.w * 0.3;
          if (b.s >= STRONG_SCORE) { best.fake = false; best.vN = 0; best.vAvg = 0; best.nextV = 0; }
          else if (moved) { best.nextV = 0; best.fakeBox = { cx: b.x + b.w / 2, cy: b.y + b.h / 2, w: b.w }; this._rememberFake(best, now); }
        }
      } else if (!this.tracks.some((t) => !t.fake && t.seen === now && coverage(t, b) > 0.3)) {
        // いま見えている顔と大きく重なる検出は、同じ顔の二重検出なので新しい顔として扱わない
        this.tracks.push({ id: this.nextId++, x: b.x, y: b.y, w: b.w, h: b.h, seen: now, clear: false, rot: b.rot || 0, view: b.view || null, altIdx: 0, s: b.s, rotAt: now, kp: b.kp, kpAt: now, person: null, nextId: 0, hits: 0, misses: 0, noAuto: false, verified: false, fake: b.s < STRONG_SCORE && this._inFakeZone(b, now), vAvg: 0, vN: 0, strong: false, nextV: 0, fakeBox: { cx: b.x + b.w / 2, cy: b.y + b.h / 2, w: b.w } });
      }
    }
    // 見失った顔の整理。ぼかさない顔は、見失ったら選択を解除する（次に現れたらぼかす）
    const W = this.out.width, H = this.out.height;
    this.tracks = this.tracks.filter((t) => {
      if (t.clear) return now - t.seen < HOLD_CLEAR_MS;
      const m = t.w * 0.25;
      const atEdge = t.x < m || t.y < m || t.x + t.w > W - m || t.y + t.h > H - m;
      if (!t.verified) return now - t.seen < HOLD_UNSURE_MS;
      return now - t.seen < (atEdge ? HOLD_EDGE_MS : HOLD_BLUR_MS);
    });
  }

  _region(t, W, H) {
    const side = t.rot === 90 || t.rot === 270;   // 横倒しの顔
    const cx = t.x + t.w / 2, cy = t.y + t.h / 2 - (t.rot ? 0 : t.h * 0.08);
    const w = t.w * (side ? EXPAND_H : EXPAND_W), h = t.h * (side ? EXPAND_W : EXPAND_H);
    let x = cx - w / 2, y = cy - h / 2;
    const x0 = Math.max(0, Math.floor(x)), y0 = Math.max(0, Math.floor(y));
    const x1 = Math.min(W, Math.ceil(x + w)), y1 = Math.min(H, Math.ceil(y + h));
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  }

  // すりガラスのような、なめらかなぼかし。
  // 手順：顔の範囲を段階的に小さくして細部を消す → 少し大きくして、ずらし重ねでならす
  //       → ふちをやわらかく消す → 元の大きさに戻して重ねる
  _blurFace(t, W, H) {
    const r = this._region(t, W, H);
    if (r.w < 2 || r.h < 2) return;
    const ratio = r.h / r.w;
    const size = (c, w) => {
      const h = Math.max(2, Math.round(w * ratio));
      if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
      return h;
    };
    // 1) 顔の範囲を小さな画面に写す（段階的に縮小。一気に縮めると映像がちらつくため）
    const MW = 72;
    const ph = size(this.pre, MW * 2);
    this.pctx.imageSmoothingEnabled = true;
    this.pctx.drawImage(this.raw, r.x, r.y, r.w, r.h, 0, 0, this.pre.width, ph);
    const mh = size(this.midA, MW); size(this.midB, MW);
    const a = this.actx, b = this.bctx;
    a.imageSmoothingEnabled = true;
    a.globalAlpha = 1;
    a.drawImage(this.pre, 0, 0, this.pre.width, ph, 0, 0, MW, mh);
    // 2) にじませる：少しずつずらした絵を重ねて平均する。幅を変えて3回くり返すと、
    //    ます目の跡が残らない、こすったようなぼかしになる
    b.globalCompositeOperation = 'source-over';
    const smear = (dst, src, d) => {
      dst.globalAlpha = 1;
      dst.drawImage(src, 0, 0);
      let n = 2;
      for (let oy = -d; oy <= d; oy += d) for (let ox = -d; ox <= d; ox += d) {
        if (!ox && !oy) continue;
        dst.globalAlpha = 1 / n++;
        dst.drawImage(src, ox, oy);
      }
      dst.globalAlpha = 1;
    };
    smear(b, this.midA, BLUR_STEPS[0]);
    smear(a, this.midB, BLUR_STEPS[1]);
    smear(b, this.midA, BLUR_STEPS[2]);
    // 3) うっすら白くして、曇りガラスのように明るくする
    b.fillStyle = `rgba(255,255,255,${BLUR_HAZE})`;
    b.fillRect(0, 0, MW, mh);
    // 4) ふちをやわらかく（だ円の外側ほど透明に）
    b.globalCompositeOperation = 'destination-in';
    b.setTransform(MW / 2, 0, 0, mh / 2, MW / 2, mh / 2);
    b.fillStyle = this._mask(b);
    b.fillRect(-1, -1, 2, 2);
    b.setTransform(1, 0, 0, 1, 0, 0);
    b.globalCompositeOperation = 'source-over';
    // 5) 録画される画面へ
    const o = this.octx;
    o.imageSmoothingEnabled = true;
    o.drawImage(this.midB, 0, 0, MW, mh, r.x, r.y, r.w, r.h);
  }

  _mask(ctx) {
    if (!this._maskGrad) {
      const g = ctx.createRadialGradient(0, 0, 0.78, 0, 0, 1);
      g.addColorStop(0, 'rgba(0,0,0,1)');
      g.addColorStop(1, 'rgba(0,0,0,0)');
      this._maskGrad = g;
    }
    return this._maskGrad;
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
      if (t.fake) continue;
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
      if (t.fake) continue;
      const r = this._region(t, W, H);
      if (px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h && r.w * r.h < hitArea) {
        hit = t; hitArea = r.w * r.h;
      }
    }
    if (hit) {
      hit.clear = !hit.clear;
      hit.hits = 0; hit.misses = 0; hit.nextId = 0;
      if (hit.clear) {
        hit.verified = true;
        // この人の顔を覚える（次の照合のタイミングで特徴をとる）
        hit.noAuto = false;
        hit.person = { id: this.nextPerson++, samples: [] };
        this.persons.push(hit.person);
      } else {
        // ぼかしに戻した人は、覚えた特徴を消す。この顔は自動では外さない
        if (hit.person) this.persons = this.persons.filter((p) => p !== hit.person);
        hit.person = null;
        hit.noAuto = true;
      }
      if (this.onToggle) this.onToggle(hit.clear);
    }
  }
}

function nearest(person, desc) {
  let min = Infinity;
  for (const s of person.samples) { const d = distance(s, desc); if (d < min) min = d; }
  return min;
}

// 2つの枠の重なりが、小さいほうの枠に占める割合
function coverage(t, b) {
  const x0 = Math.max(t.x, b.x), y0 = Math.max(t.y, b.y);
  const x1 = Math.min(t.x + t.w, b.x + b.w), y1 = Math.min(t.y + t.h, b.y + b.h);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  return inter / (Math.min(t.w * t.h, b.w * b.h) || 1);
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
