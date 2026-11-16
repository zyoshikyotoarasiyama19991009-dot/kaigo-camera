// 画面の切り替えと、タイトル一覧の管理
import { BlurCamera } from './camera.js';
import { CanvasRecorder, buildFileName, saveVideo } from './recorder.js';

const $ = (id) => document.getElementById(id);
const PRESETS = ['吸引', 'オムツ交換', '胃ろう注入', '着替え', '車いす移乗', '体位変換', '食事介助', '口腔ケア'];
const KEY = 'kaigoCamera.customTitles';

const screens = { title: $('screenTitle'), camera: $('screenCamera'), review: $('screenReview') };
function show(name) {
  for (const [k, el] of Object.entries(screens)) el.hidden = k !== name;
  document.documentElement.classList.toggle('camOn', name === 'camera');
  window.scrollTo(0, 0);
}

// ---- 画面1：内容を選ぶ ----
let custom = [];
try { custom = JSON.parse(localStorage.getItem(KEY) || '[]'); } catch (e) { custom = []; }
if (!Array.isArray(custom)) custom = [];
let selected = '';

function saveCustom() {
  try { localStorage.setItem(KEY, JSON.stringify(custom)); } catch (e) { /* 保存できなくても動かす */ }
}

function renderTitles() {
  const list = $('titleList');
  list.textContent = '';
  for (const name of [...PRESETS, ...custom.filter((c) => !PRESETS.includes(c))]) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip' + (name === selected ? ' on' : '');
    b.textContent = name;
    b.addEventListener('click', () => { selected = name; renderTitles(); });
    if (!PRESETS.includes(name)) attachLongPress(b, () => {
      custom = custom.filter((c) => c !== name);
      if (selected === name) selected = '';
      saveCustom(); renderTitles();
    });
    list.appendChild(b);
  }
  const start = $('startCamera');
  start.disabled = !selected;
  start.textContent = selected ? `「${selected}」を撮影する` : '内容を選んでください';
}

function attachLongPress(el, fn) {
  let timer = null;
  const clear = () => { clearTimeout(timer); timer = null; };
  el.addEventListener('pointerdown', () => { timer = setTimeout(() => { timer = null; fn(); }, 800); });
  for (const ev of ['pointerup', 'pointerleave', 'pointercancel']) el.addEventListener(ev, clear);
  el.addEventListener('contextmenu', (e) => e.preventDefault());
}

$('addForm').addEventListener('submit', (e) => {
  e.preventDefault();
  const v = $('addInput').value.trim();
  if (!v) return;
  if (!PRESETS.includes(v) && !custom.includes(v)) { custom.push(v); saveCustom(); }
  selected = v;
  $('addInput').value = '';
  renderTitles();
});

// ---- 画面2：カメラ ----
const camera = new BlurCamera({ video: $('cam'), out: $('out'), overlay: $('overlay') });
const recorder = new CanvasRecorder($('out'));
window.__kaigoCamera = camera; // 動作確認用
// 一度タップしたら案内文を消す（「ぼかしなし」の表示と重ならないように）
camera.onToggle = () => { $('hint').hidden = true; };
let wakeLock = null;
let timerId = null;
let result = null; // { blob, url, fileName }

function setStatus(text) {
  $('status').hidden = !text;
  $('status').textContent = text || '';
}

function labelText() {
  const d = new Date();
  return `${selected}　${d.getFullYear()}/${d.getMonth() + 1}/${d.getDate()}`;
}

$('startCamera').addEventListener('click', async () => {
  if (!selected) return;
  show('camera');
  $('recBtn').disabled = true;
  $('hint').hidden = false;
  camera.setLabel(labelText());
  camera.searchTurned = $('lyingToggle').checked;
  setStatus('カメラと顔検出を準備しています…\n（準備ができるまで画面全体がぼけます）');
  try {
    await camera.start();
    setStatus('');
    $('recBtn').disabled = false;
  } catch (e) {
    const denied = e && (e.name === 'NotAllowedError' || e.name === 'SecurityError');
    setStatus(denied
      ? 'カメラの使用が許可されていません。\nSafariの設定でカメラを許可してから、もう一度開いてください。'
      : '準備に失敗しました。\n' + (e && e.message ? e.message : e) + '\n「もどる」を押してやり直してください。');
  }
  try { wakeLock = await navigator.wakeLock?.request('screen'); } catch (e) { wakeLock = null; }
});

function leaveCamera() {
  camera.lock(false);
  camera.stop();
  clearInterval(timerId);
  $('recBadge').hidden = true;
  $('recBtn').classList.remove('on');
  try { wakeLock?.release(); } catch (e) { /* 何もしない */ }
  wakeLock = null;
}

$('backBtn').addEventListener('click', async () => {
  if (recorder.recording) {
    // 録画中に戻る＝破棄
    try { await recorder.stop(); } catch (e) { /* 何もしない */ }
  }
  leaveCamera();
  show('title');
});

$('flipBtn').addEventListener('click', async () => {
  if (recorder.recording) return;
  try { await camera.flip(); } catch (e) { setStatus('カメラを切り替えられませんでした。'); setTimeout(() => setStatus(''), 2000); }
});

$('recBtn').addEventListener('click', async () => {
  const btn = $('recBtn');
  btn.disabled = true;
  try {
    if (!recorder.recording) {
      if (!camera.ready) throw new Error('顔検出の準備ができていません');
      camera.setLabel(labelText());
      camera.lock(true);
      const { audioOk } = await recorder.start({ audio: $('audioToggle').checked });
      btn.classList.add('on');
      btn.setAttribute('aria-label', '録画停止');
      $('flipBtn').disabled = true;
      $('recBadge').hidden = false;
      $('recTime').textContent = '0:00';
      timerId = setInterval(() => {
        const s = Math.floor(recorder.elapsedMs / 1000);
        $('recTime').textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
      }, 500);
      if ($('audioToggle').checked && !audioOk) {
        setStatus('マイクが使えないため、音声なしで録画しています。');
        setTimeout(() => setStatus(''), 3000);
      }
    } else {
      const startedDate = new Date(Date.now() - recorder.elapsedMs);
      const { blob, type, durationMs } = await recorder.stop();
      btn.setAttribute('aria-label', '録画開始');
      $('flipBtn').disabled = false;
      leaveCamera();
      openReview(blob, type, startedDate, durationMs);
    }
  } catch (e) {
    camera.lock(false);
    setStatus('録画できませんでした。\n' + (e && e.message ? e.message : e));
    setTimeout(() => setStatus(''), 4000);
  } finally {
    btn.disabled = false;
  }
});

// アプリを閉じたり別の画面に切り替えたりしたら、録画中の動画は破棄する（途中までの動画を残さない）
document.addEventListener('visibilitychange', async () => {
  if (document.hidden && recorder.recording) {
    try { await recorder.stop(); } catch (e) { /* 何もしない */ }
    leaveCamera();
    show('title');
  }
});

// ---- 画面3：確認 ----
function openReview(blob, type, date, durationMs) {
  const fileName = buildFileName(selected, date, type);
  const url = URL.createObjectURL(blob);
  result = { blob, url, fileName };
  $('reviewName').textContent = fileName;
  $('review').src = url;
  const mb = (blob.size / 1024 / 1024).toFixed(1);
  const s = Math.round(durationMs / 1000);
  $('reviewInfo').textContent = `長さ 約${Math.floor(s / 60)}分${s % 60}秒 ／ ${mb} MB ／ 形式 ${type}`;
  $('saveMsg').textContent = '';
  $('againBtn').hidden = true;
  $('saveBtn').disabled = blob.size === 0;
  if (blob.size === 0) $('saveMsg').textContent = '動画が空でした。この端末では録画がうまくできていません。';
  show('review');
}

function dropResult() {
  if (!result) return;
  const v = $('review');
  v.pause(); v.removeAttribute('src'); v.load();
  URL.revokeObjectURL(result.url);
  result = null;
}

$('discardBtn').addEventListener('click', () => { dropResult(); show('title'); });
$('againBtn').addEventListener('click', () => { dropResult(); show('title'); });

$('saveBtn').addEventListener('click', async () => {
  if (!result) return;
  const how = await saveVideo(result.blob, result.fileName);
  if (how === 'shared') {
    $('saveMsg').textContent = '共有メニューで選んだ場所に保存されました。';
    $('againBtn').hidden = false;
  } else if (how === 'downloaded') {
    $('saveMsg').textContent = 'ダウンロードフォルダに保存しました。';
    $('againBtn').hidden = false;
  } else {
    $('saveMsg').textContent = '保存をやめました。もう一度「保存する」を押せます。';
  }
});

renderTitles();
show('title');
