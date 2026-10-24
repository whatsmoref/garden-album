/* Garden Photos —— macOS 照片风格前端
 * 设计约束：
 *  1. 任何图片都不做 opacity 淡入；缩略图由后端 immutable 缓存，避免“闪一下”。
 *  2. 只有点击刷新/按 ⌘R 才重新取数据；普通导航走视图缓存，瞬时返回。
 *  3. 索引/重打标完成只提示，不自动重绘，避免后台任务把正在看的网格抖一下。
 */
'use strict';

(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const clamp = (n, a, b) => Math.max(a, Math.min(b, n));

  const icon = (name, cls = '') =>
    `<svg class="ic ${cls}"><use href="#i-${name}"></use></svg>`;

  const esc = (s) =>
    String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');

  let API = '';
  let bodyEl = null;
  let scroller = null;

  const S = {
    layout: localStorage.getItem('garden.layout') || 'day',
    zoom: clamp(+localStorage.getItem('garden.zoom') || 170, 110, 280),
    theme: localStorage.getItem('garden.theme') || 'auto',
    libraryDir: '',
    onboarding: false,
    setupWaits: 0,
    stats: null,
    albums: [],
    memories: null,
    status: null,
    wasRunning: false,
    viewCache: new Map(),
    current: null,
    detailCache: new Map(),
    select: { on: false, ids: new Set(), last: null },
    tagPicked: new Set(),
    importingAlbum: null,
    toastTimer: null,
    pollTimer: null,
    lastRefresh: 0,
    typing: false,
    menu: { id: null, lbId: null, lastFocus: null, ctx: null, ctxFocus: null },
    modal: { open: false, actions: [], batchIds: null, batchTags: null },
    lb: {
      open: false,
      list: [],
      index: 0,
      currentId: null,
      info: false,
      lastFocus: null,
      token: 0,
      original: false,
      id: null,
      zoom: 1,
      pan: { x: 0, y: 0 },
    },
  };

  const VIEW_LABEL = {
    library: '图库',
    memories: '回忆',
    people: '人物',
    places: '地点',
    clean: '清理建议',
    dup: '重复项目',
    shots: '截屏',
    text: '有文字',
    bursts: '连拍',
    tags: '标签',
    tag: '标签',
    album: '相簿',
    person: '人物',
    event: '事件',
    city: '地点',
    search: '搜索',
  };

  /* ───────────────────────── 基础工具 ───────────────────────── */

  function parseDate(s) {
    if (!s) return null;
    let str = String(s).trim();
    if (/^\d{4}:\d{2}:\d{2}/.test(str)) {
      str = str.replace(/^(\d{4}):(\d{2}):(\d{2})/, '$1-$2-$3');
    }
    const d = new Date(str.replace(' ', 'T'));
    return Number.isNaN(d.getTime()) ? null : d;
  }

  const pad2 = (n) => String(n).padStart(2, '0');
  const WEEK = ['星期日', '星期一', '星期二', '星期三', '星期四', '星期五', '星期六'];

  function fmtTime(d) {
    return d ? `${pad2(d.getHours())}:${pad2(d.getMinutes())}` : '';
  }

  function fmtDayTitle(d, withYear) {
    if (!d) return '未知日期';
    const y = d.getFullYear();
    const now = new Date().getFullYear();
    if (withYear || y !== now) return `${y}年${d.getMonth() + 1}月${d.getDate()}日`;
    return `${d.getMonth() + 1}月${d.getDate()}日`;
  }

  function fmtMonthTitle(d) {
    return d ? `${d.getFullYear()}年${d.getMonth() + 1}月` : '未知日期';
  }

  function fmtFull(d) {
    if (!d) return '未知时间';
    return `${d.getFullYear()}年${d.getMonth() + 1}月${d.getDate()}日 ${WEEK[d.getDay()]} ${fmtTime(d)}`;
  }

  function fmtMonthDay(d) {
    return d ? `${d.getMonth() + 1}月${d.getDate()}日` : '';
  }

  function dateKey(d) {
    if (!d) return 'unknown';
    if (S.layout === 'year') return String(d.getFullYear());
    if (S.layout === 'month') return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}`;
    return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
  }

  function rangeLabel(photos) {
    const ds = photos.map((p) => parseDate(p.taken_at)).filter(Boolean).sort((a, b) => a - b);
    if (!ds.length) return '';
    const a = ds[0];
    const b = ds[ds.length - 1];
    if (a.getFullYear() === b.getFullYear()) {
      if (a.getMonth() === b.getMonth()) return `${a.getFullYear()}年${a.getMonth() + 1}月`;
      return `${a.getFullYear()}年${a.getMonth() + 1}月–${b.getMonth() + 1}月`;
    }
    return `${a.getFullYear()}年–${b.getFullYear()}年`;
  }

  function basename(p) {
    if (!p) return '';
    return String(p).split('/').pop();
  }

  function metaBits(bits) {
    return bits
      .filter((b) => b !== '' && b != null)
      .map((b) => `<span>${esc(b)}</span>`)
      .join('<span class="dot"></span>');
  }

  function thumbUrl(id, size = 480) {
    return `${API}/api/thumb?id=${encodeURIComponent(id)}&size=${size}`;
  }

  function faceUrl(id, size = 240) {
    return `${API}/api/face?id=${encodeURIComponent(id)}&size=${size}`;
  }

  function originalUrl(id) {
    return `${API}/api/original?id=${encodeURIComponent(id)}`;
  }

  function toast(msg, isErr = false) {
    const el = $('#toast');
    el.textContent = msg;
    el.classList.toggle('err', !!isErr);
    el.hidden = false;
    clearTimeout(S.toastTimer);
    S.toastTimer = setTimeout(() => { el.hidden = true; }, isErr ? 4200 : 2400);
  }

  function pix(n) {
    return new Intl.NumberFormat('zh-CN').format(n || 0);
  }

  /* ───────────────────────── 网络 ───────────────────────── */

  async function resolveApi() {
    try {
      const inv = window.__TAURI__ && (window.__TAURI__.core || window.__TAURI__);
      if (inv && typeof inv.invoke === 'function') {
        const base = await inv.invoke('api_base');
        if (base) API = String(base);
      }
    } catch (_) { /* 浏览器里没有 Tauri，忽略 */ }
    if (!API) {
      const p = new URLSearchParams(location.search).get('api');
      if (p) API = p;
      else if (location.protocol.startsWith('http')) API = location.origin;
      else API = 'http://127.0.0.1:8899';
    }
    API = API.replace(/\/+$/, '');
  }

  function buildUrl(path, params) {
    let url = API + path;
    if (params) {
      const qs = new URLSearchParams();
      for (const [k, v] of Object.entries(params)) {
        if (v === undefined || v === null || v === '') continue;
        qs.set(k, String(v));
      }
      const s = qs.toString();
      if (s) url += (url.includes('?') ? '&' : '?') + s;
    }
    return url;
  }

  async function getJSON(path, params) {
    const url = buildUrl(path, params);
    let lastErr = null;
    for (let attempt = 0; attempt < 7; attempt++) {
      try {
        const res = await fetch(url, { cache: 'no-store' });
        const text = await res.text();
        let data = null;
        try { data = text ? JSON.parse(text) : null; } catch (_) { data = text; }
        if (!res.ok) {
          const m = data && data.error ? data.error : `HTTP ${res.status}`;
          throw new Error(m);
        }
        if (data && data.error) throw new Error(data.error);
        return data;
      } catch (e) {
        lastErr = e;
        const network = e instanceof TypeError || /fetch|network|load failed/i.test(e.message || '');
        if (!network) throw e;
        await sleep(300 * Math.min(attempt + 1, 4));
      }
    }
    throw lastErr || new Error('连接本地服务失败');
  }

  async function postJSON(path, body) {
    const res = await fetch(API + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || (data && data.error)) {
      throw new Error((data && data.error) || `HTTP ${res.status}`);
    }
    return data;
  }

  /* ───────────────────────── 路由 ───────────────────────── */

  function normalizeRoute(r) {
    const out = { view: (r && r.view) || 'library' };
    for (const k of ['album', 'person', 'event', 'city', 'q', 'name', 'tag']) {
      if (r && r[k] !== undefined && r[k] !== null && r[k] !== '') out[k] = r[k];
    }
    return out;
  }

  function routeKey(r) {
    return [r.view, r.album || '', r.person || '', r.event || '', r.city || '', r.q || '', r.tag || ''].join('|');
  }

  function hashOf(r) {
    const p = new URLSearchParams();
    p.set('v', r.view);
    if (r.album) p.set('album', r.album);
    if (r.person) p.set('person', r.person);
    if (r.event) p.set('event', r.event);
    if (r.city) p.set('city', r.city);
    if (r.q) p.set('q', r.q);
    if (r.tag) p.set('tag', r.tag);
    if (r.name) p.set('n', r.name);
    return '#' + p.toString();
  }

  function parseHash() {
    if (!location.hash || location.hash.length < 2) return null;
    const p = new URLSearchParams(location.hash.slice(1));
    const view = p.get('v');
    if (!view) return null;
    return normalizeRoute({
      view,
      album: p.get('album') || undefined,
      person: p.get('person') || undefined,
      event: p.get('event') || undefined,
      city: p.get('city') || undefined,
      q: p.get('q') || undefined,
      tag: p.get('tag') || undefined,
      name: p.get('n') || undefined,
    });
  }

  /* ───────────────────────── 视图缓存 ───────────────────────── */

  function newState(route) {
    return {
      key: routeKey(route),
      route,
      node: null,
      photos: [],
      photoList: [],
      total: 0,
      offset: 0,
      exhausted: true,
      kind: 'grid',
      flat: false,
      items: [],
      extra: {},
      hero: { title: '', top: '', meta: '', pills: '' },
      scrollTop: 0,
      loadingMore: false,
      stale: false,
    };
  }

  function stashCurrent() {
    const st = S.current;
    if (!st || !st.node) return;
    st.scrollTop = scroller ? scroller.scrollTop : 0;
    if (st.photos.length > 800 || st.items.length > 800) return;
    S.viewCache.delete(st.key);
    S.viewCache.set(st.key, st);
    while (S.viewCache.size > 3) {
      const k = S.viewCache.keys().next().value;
      S.viewCache.delete(k);
    }
  }

  function adopt(key) {
    const st = S.viewCache.get(key);
    if (!st) return null;
    S.viewCache.delete(key);
    return st;
  }

  function mount(st) {
    if (!st.node) {
      st.node = document.createElement('div');
      st.node.className = 'body';
      st.node.innerHTML = '<div class="loading">正在读取本地索引…</div>';
    }
    if (bodyEl && bodyEl.parentNode) bodyEl.replaceWith(st.node);
    bodyEl = st.node;
  }

  async function go(route, opts = {}) {
    const r = normalizeRoute(route);
    const key = routeKey(r);
    const historyMode = opts.historyMode || 'push';

    if (!opts.force && S.current && S.current.key === key) {
      if (historyMode === 'push') {
        try { history.pushState({}, '', hashOf(r)); } catch (_) {}
      }
      return;
    }

    closeLightbox();
    hidePopover();
    hidePhotoMenu();
    hideCtxMenu();
    if (S.select.on) setSelectMode(false);
    stashCurrent();
    S.current = null;
    if (opts.force) S.viewCache.delete(key);

    if (historyMode === 'push') {
      try { history.pushState({}, '', hashOf(r)); } catch (_) {}
    } else if (historyMode === 'replace') {
      try { history.replaceState({}, '', hashOf(r)); } catch (_) {}
    }

    updateNav(r);
    $('#btnBack').style.visibility = r.view === 'library' ? 'hidden' : 'visible';
    if (r.view !== 'search') {
      const q = $('#q');
      if (q && q.value) q.value = '';
      $('#chips').innerHTML = '';
      $('#searchNote').textContent = '';
      $('#searchNote').classList.remove('err');
    }

    const cached = opts.force ? null : adopt(key);
    if (cached) {
      S.current = cached;
      mount(cached);
      applyHero(cached.hero);
      requestAnimationFrame(() => {
        if (scroller) scroller.scrollTop = cached.scrollTop || 0;
      });
      updateSegState();
      return;
    }

    const st = newState(r);
    S.current = st;
    mount(st);
    applyHero(loadingHero(r));
    updateSegState();
    await loadRoute(st);
  }

  function loadingHero(r) {
    const label = r.name || VIEW_LABEL[r.view] || '照片';
    return { title: label, top: label, meta: '<span>正在载入…</span>', pills: '' };
  }

  function applyHero(hero) {
    if (!hero) return;
    $('#pageTitle').textContent = hero.title || '';
    $('#pageMeta').innerHTML = hero.meta || '';
    $('#pills').innerHTML = hero.pills || '';
    $('#topTitle').textContent = hero.top || hero.title || '';
    document.title = `${hero.top || hero.title || '照片'} · Garden`;
  }

  function updateNav(route) {
    $$('.nav-item').forEach((el) => el.classList.remove('active'));
    const map = {
      library: 'library',
      memories: 'memories',
      event: 'memories',
      people: 'people',
      person: 'people',
      places: 'places',
      city: 'places',
      clean: 'clean',
      dup: 'dup',
      shots: 'shots',
      text: 'text',
      bursts: 'bursts',
      tags: 'tags',
      tag: 'tags',
      search: 'library',
    };
    if (route.view === 'album') {
      const el = $(`.album-item[data-album="${String(route.album).replace(/"/g, '')}"]`);
      if (el) el.classList.add('active');
    } else {
      const v = map[route.view];
      const el = v ? $(`.nav-item[data-view="${v}"]`) : null;
      if (el) el.classList.add('active');
    }
  }

  /* ───────────────────────── 侧栏 / 统计 ───────────────────────── */

  function setCount(id, n) {
    const el = document.getElementById(id);
    if (el) el.textContent = n ? pix(n) : '';
  }

  async function loadChrome() {
    const jobs = [
      getJSON('/api/stats').then((s) => {
        S.stats = s;
        setCount('cLibrary', s.photos);
        setCount('cMemories', s.memories != null ? s.memories : s.events);
        setCount('cPeople', s.persons);
        if (s.places != null) setCount('cPlaces', s.places);
        setCount('cShots', s.screenshots);
        setCount('cText', s.with_ocr);
        setCount('cBursts', s.bursts);
        setCount('cTags', s.tags);
        updateFooter();
      }).catch(() => {}),
      getJSON('/api/albums').then((d) => {
        S.albums = d.albums || [];
        renderAlbumNav();
        updateNav(S.current ? S.current.route : { view: 'library' });
      }).catch(() => {}),
    ];
    await Promise.allSettled(jobs);
  }

  function renderAlbumNav() {
    const wrap = $('#albumNav');
    const albums = S.albums.filter((a) => a.kind === 'manual' || (a.photo_count || 0) > 0);
    if (!albums.length) {
      wrap.innerHTML = '<div class="nav-title" style="text-transform:none;letter-spacing:0;font-weight:400;padding-left:9px">暂无相簿</div>';
      return;
    }
    wrap.innerHTML = albums.map((a) => `
      <button class="nav-item album-item" data-album="${esc(a.id)}" data-name="${esc(a.name)}">
        ${icon('folder')}
        <span>${esc(a.name)}</span>
        <em>${a.photo_count ? pix(a.photo_count) : ''}</em>
      </button>`).join('');
  }

  function updateFooter() {
    const st = S.status;
    const left = $('#statusLeft');
    const right = $('#statusRight');
    if (st && st.running) {
      const pct = st.total ? ` ${pix(st.done)}/${pix(st.total)}` : '';
      left.textContent = `${st.kind === 'retag' ? '正在重建智能标签' : st.kind === 'import' ? '正在导入文件夹' : st.kind === 'setup' ? '正在建立照片库' : '正在扫描照片'}${pct}`;
      right.textContent = st.phase === 'loading' ? '准备中…' : '后台任务进行中';
      $('#pulse').classList.add('busy');
      $('#footText').textContent = left.textContent;
    } else {
      $('#pulse').classList.remove('busy');
      if (S.stats) {
        left.textContent = `${pix(S.stats.photos)} 张照片 · ${pix(S.stats.tags)} 个标签 · ${pix(S.stats.persons)} 人`;
        $('#footText').textContent = '本地索引已就绪';
      } else {
        $('#footText').textContent = '正在连接本地索引…';
      }
      if (S.lastRefresh) {
        right.textContent = `刷新于 ${fmtTime(new Date(S.lastRefresh))}`;
      } else {
        right.textContent = '';
      }
    }
  }

  async function pollStatus() {
    clearTimeout(S.pollTimer);
    try {
      const st = await getJSON('/api/index/status');
      S.status = st;
      updateFooter();
      if (st.running) {
        S.wasRunning = true;
      } else if (S.wasRunning) {
        S.wasRunning = false;
        const wasImport = !!S.importingAlbum || st.kind === 'import';
        S.importingAlbum = null;
        toast(st.error ? `任务失败：${st.error}`
          : wasImport ? '文件夹导入完成 · 点刷新查看新照片'
          : '索引完成 · 点刷新查看最新内容', !!st.error);
        loadChrome();
        if (wasImport) await syncAlbums().catch(() => {});
      }
    } catch (_) { /* 后端尚未就绪 */ }
    S.pollTimer = setTimeout(pollStatus, 3000);
  }

  /* ───────────────────────── 加载各视图 ───────────────────────── */

  async function loadRoute(st) {
    const v = st.route.view;
    try {
      if (['library', 'event', 'person', 'album', 'city', 'shots', 'tag'].includes(v)) {
        await loadGrid(st);
      } else if (v === 'text') {
        await loadText(st);
      } else if (v === 'search') {
        await loadSearch(st);
      } else if (v === 'memories' || v === 'places') {
        await loadMemories(st, v === 'places');
      } else if (v === 'people') {
        await loadPeople(st);
      } else if (v === 'clean' || v === 'dup') {
        await loadClean(st, v === 'dup');
      } else if (v === 'bursts') {
        await loadBursts(st);
      } else if (v === 'tags') {
        await loadTags(st);
      } else {
        await loadGrid(st);
      }
    } catch (e) {
      if (S.current !== st) return;
      st.node.innerHTML = errorHTML(e.message);
      applyHero({ title: VIEW_LABEL[v] || '照片', top: VIEW_LABEL[v] || '照片', meta: '<span>加载失败</span>', pills: '' });
    }
  }

  function gridParams(route) {
    const p = { limit: 1000, offset: 0, summary: 1 };
    if (route.view === 'event') p.event = route.event;
    if (route.view === 'person') p.person = route.person;
    if (route.view === 'album') p.album = route.album;
    if (route.view === 'city') p.city = route.city;
    if (route.view === 'shots') p.screenshot = 1;
    if (route.view === 'tag') p.tag = route.tag;
    return p;
  }

  async function loadGrid(st) {
    const data = await getJSON('/api/photos', gridParams(st.route));
    if (S.current !== st) return;
    st.kind = 'grid';
    st.photos = data.photos || [];
    st.total = data.total || st.photos.length;
    st.offset = st.photos.length;
    st.exhausted = st.photos.length >= st.total;
    st.flat = false;
    st.photoList = st.photos.map((p) => p.id);
    st.hero = heroForGrid(st);
    renderGrid(st);
    applyHero(st.hero);
  }

  async function loadText(st) {
    const p = { limit: 1000, offset: 0, summary: 0, ocr: 1 };
    const data = await getJSON('/api/photos', p);
    if (S.current !== st) return;
    st.kind = 'rows';
    st.photos = data.photos || [];
    st.total = data.total || st.photos.length;
    st.exhausted = true;
    st.photoList = st.photos.map((x) => x.id);
    st.hero = {
      title: '有文字',
      top: '有文字',
      meta: metaBits([`${pix(st.total)} 张含文字照片`, 'OCR 在本机识别']),
      pills: '',
    };
    renderTextRows(st);
    applyHero(st.hero);
  }

  async function loadSearch(st) {
    const data = await getJSON('/api/search', { q: st.route.q, n: 300 });
    if (S.current !== st) return;
    st.kind = 'grid';
    st.flat = true;
    st.photos = data.results || [];
    st.total = st.photos.length;
    st.exhausted = true;
    st.photoList = st.photos.map((p) => p.id);
    st.extra.elapsed = data.elapsed_ms;
    st.extra.chips = data.chips || [];
    const chipsHTML = (data.chips || [])
      .map(([k, t]) => `<span class="cchip k-${esc(k)}">${esc(t)}</span>`)
      .join('');
    $('#chips').innerHTML = chipsHTML;
    const note = $('#searchNote');
    note.classList.remove('err');
    note.textContent = `找到 ${pix(st.total)} 个结果 · ${data.elapsed_ms == null ? '' : data.elapsed_ms + ' ms'}`;
    st.hero = {
      title: `“${st.route.q}”`,
      top: '搜索',
      meta: metaBits([`${pix(st.total)} 个结果`, data.elapsed_ms != null ? `${data.elapsed_ms} ms` : '', '本地语义检索']),
      pills: `<button class="pill" data-action="clearSearch">${icon('close')}清除搜索</button>`,
    };
    renderGrid(st);
    applyHero(st.hero);
  }

  async function loadMemories(st, places) {
    const data = await getJSON('/api/memories', { limit: 60, per_event: 10 });
    if (S.current !== st) return;
    const events = (data.events || []).filter((e) => (e.photos || []).length && (places ? !!e.city : (e.photo_count || 0) >= 3));
    st.extra.events = events;
    S.memories = events;
    st.kind = places ? 'places' : 'memories';
    st.items = events;
    if (places) {
      const cities = new Set(events.map((e) => e.city).filter(Boolean));
      setCount('cPlaces', cities.size);
      st.hero = {
        title: '地点',
        top: '地点',
        meta: metaBits([`${pix(cities.size)} 个城市`, `${pix(events.length)} 段行程`, '按拍摄地点聚合']),
        pills: '',
      };
      renderPlaces(st);
    } else {
      setCount('cMemories', events.length);
      st.hero = {
        title: '回忆',
        top: '回忆',
        meta: metaBits([`${pix(events.length)} 段回忆`, '自动按时间与地点整理']),
        pills: '',
      };
      renderMemories(st);
    }
    applyHero(st.hero);
  }

  async function loadPeople(st) {
    const data = await getJSON('/api/persons');
    if (S.current !== st) return;
    const persons = (data.persons || []).filter((p) => p.photos > 0);
    st.kind = 'people';
    st.items = persons;
    setCount('cPeople', persons.length);
    st.hero = {
      title: '人物',
      top: '人物',
      meta: metaBits([`${pix(persons.length)} 个人物`, `${pix(persons.reduce((a, b) => a + (b.photos || 0), 0))} 张出镜照片`, '本机人脸聚类']),
      pills: '',
    };
    renderPeople(st);
    applyHero(st.hero);
  }

  async function loadClean(st, dupOnly) {
    const data = await getJSON('/api/cleanup');
    if (S.current !== st) return;
    let items = data.items || [];
    if (dupOnly) items = items.filter((it) => it.type === 'duplicate');
    st.kind = 'rows';
    st.items = items;
    st.photoList = items.map((it) => it.photo && it.photo.id).filter(Boolean);
    const counts = {};
    for (const it of items) counts[it.type] = (counts[it.type] || 0) + 1;
    const pills = Object.entries(counts)
      .map(([t, n]) => `<span class="pill p-clean">${esc(cleanLabel(t))} ${pix(n)}</span>`)
      .join('');
    st.hero = {
      title: dupOnly ? '重复项目' : '清理建议',
      top: dupOnly ? '重复项目' : '清理建议',
      meta: metaBits([`${pix(items.length)} 项建议`, '只读分析，不会自动删除']),
      pills,
    };
    renderClean(st, dupOnly);
    applyHero(st.hero);
  }

  async function loadBursts(st) {
    const data = await getJSON('/api/bursts');
    if (S.current !== st) return;
    const bursts = (data.bursts || []).filter((b) => b.best_id);
    st.kind = 'rows';
    st.items = bursts;
    st.photoList = bursts.map((b) => b.best_id);
    st.hero = {
      title: '连拍',
      top: '连拍',
      meta: metaBits([`${pix(bursts.length)} 组连拍`, '每组保留评分最高的一张']),
      pills: '',
    };
    renderBursts(st);
    applyHero(st.hero);
  }

  async function loadTags(st) {
    const data = await getJSON('/api/tags', { limit: 5000 });
    if (S.current !== st) return;
    st.kind = 'tags';
    st.items = data.tags || [];
    st.total = data.total || st.items.length;
    st.hero = {
      title: '标签',
      top: '标签',
      meta: metaBits([`${pix(st.total)} 个标签`, `${pix(st.items.reduce((a, b) => a + (b.n || 0), 0))} 个标注`, '可删除无用标签']),
      pills: '',
    };
    renderTags(st);
    applyHero(st.hero);
  }

  function renderTags(st) {
    const items = st.items || [];
    if (!items.length) {
      st.node.innerHTML = emptyHTML('还没有标签，先扫描照片或手动添加');
      return;
    }
    const picked = S.tagPicked;
    const rows = items.map((t) => {
      const zh = t.zh && t.zh !== t.tag ? t.zh : '';
      const search = `${t.tag || ''} ${zh}`.toLowerCase();
      const on = picked.has(t.tag);
      return `<div class="tag-admin${on ? ' picked' : ''}" data-action="tagView" data-tag="${esc(t.tag)}" data-name="${esc(zh || t.tag)}" data-search="${esc(search)}">
        <button type="button" class="tag-check" data-tag="${esc(t.tag)}" role="checkbox" aria-checked="${on ? 'true' : 'false'}" aria-label="选择标签 ${esc(t.tag)}">${icon('check')}</button>
        <span class="tag-name">${esc(zh || t.tag)}</span>
        ${zh ? `<span class="tag-en">${esc(t.tag)}</span>` : ''}
        ${t.manual ? `<span class="manual">手动 ${pix(t.manual)}</span>` : ''}
        <span class="n">${pix(t.n)}</span>
        <button class="del" data-action="tagDelete" data-tag="${esc(t.tag)}" title="删除标签">×</button>
      </div>`;
    }).join('');
    st.node.innerHTML = `<div class="tag-manager">
      <input class="tag-filter" id="tagFilter" placeholder="筛选标签（中文或英文）" autocomplete="off">
      <div class="tag-bulk">
        <em id="tagPickedCount">${picked.size} 个标签已选</em>
        <button data-action="tagAll">全选当前</button>
        <button data-action="tagClear">清空</button>
        <button class="primary" data-action="tagMerge" ${picked.size ? '' : 'disabled'}>合并为相簿</button>
        <button class="primary" data-action="tagToAlbum" ${picked.size ? '' : 'disabled'}>加入已有相簿</button>
      </div>
      <div class="tag-admin-grid" id="tagAdminGrid">${rows}</div>
      <p class="tag-manager-note">点标签查看该标签下的照片；勾选多个标签后可合并成一个相簿。删除会从所有照片移除该标签，并在重建标签时不再出现。</p>
    </div>`;
  }

  function toggleTagPick(tag, on) {
    if (!tag) return;
    if (on) S.tagPicked.add(tag); else S.tagPicked.delete(tag);
    const row = S.current && $$('.tag-admin', S.current.node).find((el) => el.dataset.tag === tag);
    if (row) {
      row.classList.toggle('picked', !!on);
      const chk = $('.tag-check', row);
      if (chk) chk.setAttribute('aria-checked', on ? 'true' : 'false');
    }
    updateTagBulk();
  }

  function updateTagBulk() {
    const node = S.current && S.current.node;
    if (!node) return;
    const count = S.tagPicked.size;
    const em = $('#tagPickedCount', node);
    if (em) em.textContent = `${count} 个标签已选`;
    const merge = $('[data-action="tagMerge"]', node);
    if (merge) merge.disabled = !count;
    const add = $('[data-action="tagToAlbum"]', node);
    if (add) add.disabled = !count;
    $$('.tag-admin', node).forEach((row) => {
      const on = S.tagPicked.has(row.dataset.tag);
      row.classList.toggle('picked', on);
      const chk = $('.tag-check', row);
      if (chk) chk.setAttribute('aria-checked', on ? 'true' : 'false');
    });
  }

  function filterTags(input) {
    const q = String(input && input.value || '').trim().toLowerCase();
    $$('.tag-admin', S.current && S.current.node).forEach((el) => {
      el.hidden = !!q && !(el.dataset.search || '').includes(q);
    });
  }

  function cleanLabel(t) {
    return { duplicate: '重复', burst_redundant: '连拍冗余', blurry: '模糊', low_quality: '低质量' }[t] || t;
  }

  /* ───────────────────────── Hero 区 ───────────────────────── */

  function heroForGrid(st) {
    const r = st.route;
    const n = st.total;
    const range = rangeLabel(st.photos);
    const metaCount = `${pix(n)} 张照片`;
    if (r.view === 'library') {
      const pills = [];
      if (S.stats) {
        pills.push(`<button class="pill p-memory" data-action="nav" data-nav="memories">${icon('memories')}回忆 ${pix(S.stats.memories != null ? S.stats.memories : S.stats.events)}</button>`);
        pills.push(`<button class="pill p-people" data-action="nav" data-nav="people">${icon('people')}人物 ${pix(S.stats.persons)}</button>`);
        pills.push(`<button class="pill p-place" data-action="nav" data-nav="places">${icon('place')}地点 ${S.stats.places != null ? pix(S.stats.places) : ''}</button>`);
      }
      return {
        title: '图库',
        top: '图库',
        meta: metaBits([metaCount, range]),
        pills: pills.join(''),
      };
    }
    if (r.view === 'event') {
      return {
        title: r.name || '事件',
        top: '事件',
        meta: metaBits([metaCount, range, r.city || '']),
        pills: '',
      };
    }
    if (r.view === 'person') {
      return {
        title: r.name || '人物',
        top: '人物',
        meta: metaBits([metaCount, range]),
        pills: `<button class="pill p-edit" data-action="renamePerson" data-id="${esc(r.person)}">${icon('tag')}重命名</button>`,
      };
    }
    if (r.view === 'album') {
      const a = S.albums.find((x) => String(x.id) === String(r.album));
      const pills = a && a.kind === 'manual'
        ? `<button class="pill p-edit" data-action="goLibrary">${icon('plus')}去图库添加照片</button>
           <button class="pill p-edit" data-action="renameAlbum" data-id="${esc(a.id)}">${icon('tag')}重命名</button>
           <button class="pill p-danger" data-action="deleteAlbum" data-id="${esc(a.id)}">${icon('close')}删除相簿</button>`
        : '';
      return {
        title: r.name || '相簿',
        top: '相簿',
        meta: metaBits([metaCount, range]),
        pills,
      };
    }
    if (r.view === 'city') {
      return {
        title: r.city || '地点',
        top: '地点',
        meta: metaBits([metaCount, range]),
        pills: '',
      };
    }
    if (r.view === 'shots') {
      return {
        title: '截屏',
        top: '截屏',
        meta: metaBits([metaCount, range]),
        pills: '',
      };
    }
    if (r.view === 'tag') {
      return {
        title: r.name || r.tag,
        top: '标签',
        meta: metaBits([metaCount, range, `标签：${r.tag}`]),
        pills: `<button class="pill" data-action="nav" data-nav="tags">${icon('tag')}所有标签</button>`,
      };
    }
    return { title: VIEW_LABEL[r.view] || '照片', top: VIEW_LABEL[r.view] || '照片', meta: metaBits([metaCount, range]), pills: '' };
  }

  /* ───────────────────────── 网格 ───────────────────────── */

  function tileHTML(p) {
    const name = p.filename || basename(p.path);
    const d = parseDate(p.taken_at);
    const badges = [];
    if (p.is_screenshot) badges.push(`<span class="badge">${icon('shot')}截屏</span>`);
    if (p.people_count) badges.push(`<span class="badge">${icon('people')}${p.people_count}</span>`);
    if (p.has_ocr) badges.push(`<span class="badge">${icon('text')}文字</span>`);
    const picked = S.select.ids.has(String(p.id));
    return `<div class="tile${picked ? ' selected' : ''}" data-action="photo" data-id="${esc(p.id)}" tabindex="0" role="button" aria-label="${esc(name)}" aria-selected="${picked ? 'true' : 'false'}">
      <img src="${thumbUrl(p.id, 480)}" alt="" loading="lazy" decoding="async">
      <span class="tile-check">${picked ? icon('check') : ''}</span>
      ${badges.length ? `<div class="badges">${badges.join('')}</div>` : ''}
      <div class="cap"><span class="name">${esc(name)}</span><span class="time">${fmtTime(d)}</span></div>
    </div>`;
  }

  function renderGrid(st) {
    const photos = st.photos;
    if (!photos.length) {
      const manualAlbum = st.route.view === 'album'
        && S.albums.some((a) => String(a.id) === String(st.route.album) && a.kind === 'manual');
      const importing = st.route.view === 'album' && String(S.importingAlbum) === String(st.route.album);
      st.node.innerHTML = emptyHTML(importing
        ? '正在从文件夹导入照片，完成后点刷新即可看到。'
        : manualAlbum
          ? '这个相簿还没有照片。打开任意照片，点右上角相簿按钮加入；或点上方“去图库添加照片”。'
          : st.route.view === 'library'
            ? (S.status && S.status.running && S.status.kind === 'setup'
              ? '正在建立照片库，完成后点刷新查看。'
              : '照片库还是空的，先选择照片文件夹吧')
            : st.route.view === 'tag'
              ? `标签「${st.route.name || st.route.tag}」下还没有照片`
              : '这里还没有照片', st.route.view === 'library' && !importing
                && !(S.status && S.status.running));
      return;
    }

    let html = '';
    if (st.flat || S.layout === 'all') {
      html = `<div class="grid">${photos.map(tileHTML).join('')}</div>`;
    } else {
      const groups = [];
      let curGroup = null;
      for (const p of photos) {
        const d = parseDate(p.taken_at);
        const k = dateKey(d);
        if (!curGroup || curGroup.key !== k) {
          curGroup = { key: k, date: d, items: [] };
          groups.push(curGroup);
        }
        curGroup.items.push(p);
      }
      html = groups.map((g) => {
        let title = fmtDayTitle(g.date, false);
        let sub = g.date ? WEEK[g.date.getDay()] : '';
        if (S.layout === 'year') { title = g.date ? `${g.date.getFullYear()}年` : '未知日期'; sub = ''; }
        else if (S.layout === 'month') { title = fmtMonthTitle(g.date); sub = ''; }
        const cities = [...new Set(g.items.map((p) => p.city).filter(Boolean))];
        if (S.layout === 'day' && cities.length === 1 && !sub) sub = cities[0];
        else if (S.layout === 'day' && cities.length === 1) sub += ` · ${cities[0]}`;
        return `<section class="date-group">
          <div class="date-head">
            <span class="date-title">${esc(title)}</span>
            ${sub ? `<span class="date-sub">${esc(sub)}</span>` : ''}
            <span class="date-count">${pix(g.items.length)} 张</span>
          </div>
          <div class="grid">${g.items.map(tileHTML).join('')}</div>
        </section>`;
      }).join('');
    }

    if (!st.exhausted) {
      html += `<div class="load-more-wrap"><button class="load-more" data-action="more">加载更多（已显示 ${pix(photos.length)} / ${pix(st.total)}）</button></div>`;
    }
    st.node.innerHTML = html;
  }

  function renderTextRows(st) {
    if (!st.photos.length) {
      st.node.innerHTML = emptyHTML('还没有识别到文字');
      return;
    }
    st.node.innerHTML = `<div class="rows">${st.photos.map((p) => {
      const txt = (p.ocr_text || '').replace(/\s+/g, ' ').trim();
      return photoRow(p, { tag: 'OCR', tagCls: 'info', sub: txt ? txt.slice(0, 160) : '含文字' });
    }).join('')}</div>`;
  }

  function photoRow(p, opts = {}) {
    const name = p.filename || basename(p.path);
    const d = parseDate(p.taken_at);
    const sub = opts.sub || [fmtFull(d), p.width && p.height ? `${p.width}×${p.height}` : ''].filter(Boolean).join(' · ');
    return `<div class="row" data-action="photo" data-id="${esc(p.id)}" tabindex="0" role="button">
      <img src="${thumbUrl(p.id, 480)}" alt="" loading="lazy" decoding="async">
      <div class="rb">
        <div class="rt">${esc(name)}</div>
        <div class="rs">${esc(sub)}</div>
      </div>
      ${opts.tag ? `<span class="rk ${opts.tagCls || ''}">${esc(opts.tag)}</span>` : ''}
    </div>`;
  }

  function renderClean(st, dupOnly) {
    if (!st.items.length) {
      st.node.innerHTML = emptyHTML(dupOnly ? '没有发现重复照片' : '没有需要清理的照片');
      return;
    }
    st.node.innerHTML = `<div class="rows">${st.items.map((it) => {
      const p = it.photo || {};
      const reason = String(it.reason || '').replace(/\s+/g, ' ');
      return photoRow(p, {
        tag: cleanLabel(it.type),
        tagCls: it.type === 'duplicate' ? 'info' : '',
        sub: `${reason} · 置信度 ${Math.round((it.confidence || 0) * 100)}%`,
      });
    }).join('')}</div>`;
  }

  function renderBursts(st) {
    if (!st.items.length) {
      st.node.innerHTML = emptyHTML('还没有检测到连拍');
      return;
    }
    st.node.innerHTML = `<div class="rows">${st.items.map((b) => {
      const p = { id: b.best_id, filename: basename(b.best_path), path: b.best_path };
      return photoRow(p, { tag: `${b.photo_count} 张`, tagCls: 'keep', sub: '已保留评分最高的一张' });
    }).join('')}</div>`;
  }

  /* ───────────────────────── 回忆 / 地点 / 人物 ───────────────────────── */

  function coverOf(event) {
    const photos = event.photos || [];
    if (!photos.length) return null;
    return photos.slice().sort((a, b) => (b.aesthetic || 0) - (a.aesthetic || 0))[0];
  }

  function eventRange(e) {
    const a = parseDate(e.start);
    const b = parseDate(e.end);
    if (!a) return '';
    if (!b || (a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate())) {
      return fmtMonthDay(a);
    }
    return `${fmtMonthDay(a)} – ${fmtMonthDay(b)}`;
  }

  function renderMemories(st) {
    if (!st.items.length) {
      st.node.innerHTML = emptyHTML('还没有形成回忆，导入更多照片后会自动整理');
      return;
    }
    st.node.innerHTML = `<div class="memories">${st.items.map((e) => {
      const cover = coverOf(e);
      const tag = (e.top_tags_zh || [])[0] || '回忆';
      return `<article class="memory" data-action="event" data-id="${esc(e.id)}" data-name="${esc(e.title || '')}" data-city="${esc(e.city || '')}" tabindex="0" role="button">
        <div class="memory-cover">
          ${cover ? `<img src="${thumbUrl(cover.id, 640)}" alt="" loading="lazy" decoding="async">` : ''}
        </div>
        <span class="memory-badge">${icon('spark')}${esc(tag)}</span>
        <div class="memory-info">
          <div class="memory-title">${esc(e.title || '未命名回忆')}</div>
          <div class="memory-sub">
            <span>${esc(eventRange(e))}</span>
            ${e.city ? `<span>${esc(e.city)}</span>` : ''}
            <span>${pix(e.photo_count || (e.photos || []).length)} 张</span>
          </div>
        </div>
      </article>`;
    }).join('')}</div>`;
  }

  function renderPlaces(st) {
    const events = st.items;
    const cities = new Map();
    for (const e of events) {
      if (!e.city) continue;
      if (!cities.has(e.city)) cities.set(e.city, []);
      cities.get(e.city).push(e);
    }
    if (!cities.size) {
      st.node.innerHTML = emptyHTML('还没有带 GPS 或地点信息的照片');
      return;
    }
    const sorted = [...cities.entries()].sort((a, b) => b[1].length - a[1].length);
    st.node.innerHTML = sorted.map(([city, list]) => {
      const total = list.reduce((a, e) => a + (e.photo_count || 0), 0);
      return `<section class="place-group">
        <div class="place-head">
          <span class="place-name">${esc(city)}</span>
          <span class="place-meta">${pix(list.length)} 段行程 · ${pix(total)} 张照片</span>
        </div>
        <div class="place-events">${list.map((e) => {
          const cover = coverOf(e);
          return `<article class="place-card" data-action="event" data-id="${esc(e.id)}" data-name="${esc(e.title || '')}" data-city="${esc(e.city || '')}" tabindex="0" role="button">
            ${cover ? `<img src="${thumbUrl(cover.id, 480)}" alt="" loading="lazy" decoding="async">` : ''}
            <div class="pi">
              <div class="pt">${esc(e.title || '未命名行程')}</div>
              <div class="ps">${esc(eventRange(e))} · ${pix(e.photo_count || 0)} 张</div>
            </div>
          </article>`;
        }).join('')}</div>
      </section>`;
    }).join('');
  }

  function renderPeople(st) {
    if (!st.items.length) {
      st.node.innerHTML = emptyHTML('还没有聚类出人物');
      return;
    }
    st.node.innerHTML = `<div class="people">${st.items.map((p) => {
      const auto = /^人物\d+$/.test(p.name || '');
      const initial = (p.name || '?').replace(/^人物\s*/, '').slice(0, 1) || '?';
      return `<article class="person" data-action="person" data-id="${esc(p.id)}" data-name="${esc(p.name || '')}" tabindex="0" role="button">
        <div class="person-av">${p.cover_id ? `<img src="${faceUrl(p.cover_id, 240)}" alt="" loading="lazy" decoding="async">` : esc(initial)}</div>
        <div class="person-name">${esc(p.name || '未命名')}${auto ? '<span class="auto">自动</span>' : ''}</div>
        <div class="person-count">${pix(p.photos)} 张照片</div>
      </article>`;
    }).join('')}</div>`;
  }

  /* ───────────────────────── 空 / 错误 ───────────────────────── */

  function emptyHTML(text, withScan = false) {
    return `<div class="empty">
      <div>${esc(text)}</div>
      ${withScan ? `<div style="margin-top:14px"><button class="load-more" data-action="scan">选择照片文件夹</button></div>` : ''}
    </div>`;
  }

  function errorHTML(msg) {
    return `<div class="empty">
      <div>加载失败：${esc(msg || '未知错误')}</div>
      <div style="margin-top:14px"><button class="load-more" data-action="retry">重试</button></div>
    </div>`;
  }

  /* ───────────────────────── 灯箱 ───────────────────────── */

  function findPhoto(id) {
    const st = S.current;
    if (!st) return null;
    return st.photos.find((p) => String(p.id) === String(id)) || null;
  }

  function openLightbox(id) {
    hidePhotoMenu();
    const st = S.current;
    let list = st && st.photoList && st.photoList.length ? st.photoList.slice() : [id];
    let idx = list.findIndex((x) => String(x) === String(id));
    if (idx < 0) { list = [id, ...list]; idx = 0; }
    S.lb.list = list;
    S.lb.index = idx;
    S.lb.lastFocus = document.activeElement;
    S.lb.info = false;
    S.lb.original = false;
    $('#lightbox').hidden = false;
    S.lb.open = true;
    $('#lbInfo').hidden = true;
    $('#lbInfoBtn').classList.remove('on');
    setLbZoom(1, true);
    showPhoto(idx);
    setTimeout(() => $('#lbClose').focus(), 0);
  }

  function showPhoto(idx) {
    const id = S.lb.list[idx];
    if (id === undefined) return;
    const token = ++S.lb.token;
    S.lb.index = idx;
    S.lb.currentId = id;
    S.lb.original = false;
    setLbZoom(1, true);
    const p = findPhoto(id) || S.detailCache.get(String(id)) || {};
    $('#lbTitle').textContent = p.filename || basename(p.path) || `照片 #${id}`;
    const d = parseDate(p.taken_at);
    $('#lbCounter').textContent = `${idx + 1} / ${S.lb.list.length}${d ? ' · ' + fmtFull(d) : ''}`;
    $('#lbLoading').hidden = false;

    loadLbImage(thumbUrl(id, 2048), token, () => {
      loadLbImage(thumbUrl(id, 1024), token, () => {
        if (token === S.lb.token) {
          $('#lbLoading').hidden = true;
          toast('这张照片暂时无法预览', true);
        }
      });
    });

    for (const n of [idx - 1, idx + 1]) {
      const nid = S.lb.list[n];
      if (nid !== undefined) {
        const im = new Image();
        im.src = thumbUrl(nid, 2048);
      }
    }

    if (S.lb.info) fillInfo(id);
    else $('#lbInfo').innerHTML = '<div class="loading">正在读取信息…</div>';
  }

  function loadLbImage(src, token, onFail) {
    const pre = new Image();
    pre.onload = () => {
      if (token !== S.lb.token) return;
      const img = $('#lbImg');
      img.src = pre.src;
      img.classList.add('ready');
      $('#lbLoading').hidden = true;
    };
    pre.onerror = () => {
      if (token !== S.lb.token) return;
      if (onFail) onFail();
    };
    pre.src = src;
  }

  function setLbZoom(z, reset = false) {
    const zoom = clamp(Number(z) || 1, .12, 8);
    S.lb.zoom = zoom;
    if (reset) S.lb.pan = { x: 0, y: 0 };
    const img = $('#lbImg');
    if (img) {
      const css = zoom === 1 ? '' : `translate3d(${S.lb.pan.x}px, ${S.lb.pan.y}px, 0) scale(${zoom})`;
      img.style.transform = css;
    }
    const stage = $('#lbStage');
    if (stage) stage.classList.toggle('zoomed', zoom > 1);
    const label = $('#lbZoomLabel');
    if (label) label.textContent = zoom === 1 ? '适应' : `${Math.round(zoom * 100)}%`;
  }

  function changeLbZoom(factor) {
    if (!S.lb.open) return;
    setLbZoom(S.lb.zoom * factor);
  }

  function bindLbPan() {
    const stage = $('#lbStage');
    const img = $('#lbImg');
    let drag = null;
    stage.addEventListener('pointerdown', (e) => {
      if (!S.lb.open || S.lb.zoom <= 1 || e.button !== 0 || e.target.closest('#lbLoading')) return;
      drag = { id: e.pointerId, x: e.clientX, y: e.clientY, px: S.lb.pan.x, py: S.lb.pan.y };
      stage.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    stage.addEventListener('pointermove', (e) => {
      if (!drag || drag.id !== e.pointerId) return;
      S.lb.pan.x = drag.px + e.clientX - drag.x;
      S.lb.pan.y = drag.py + e.clientY - drag.y;
      const imgEl = $('#lbImg');
      if (imgEl) imgEl.style.transform = `translate3d(${S.lb.pan.x}px, ${S.lb.pan.y}px, 0) scale(${S.lb.zoom})`;
    });
    const stop = (e) => {
      if (drag && drag.id === e.pointerId) {
        try { stage.releasePointerCapture(e.pointerId); } catch (_) {}
        drag = null;
      }
    };
    stage.addEventListener('pointerup', stop);
    stage.addEventListener('pointercancel', stop);
    img.addEventListener('wheel', (e) => {
      if (!S.lb.open) return;
      e.preventDefault();
      const next = e.deltaY > 0 ? S.lb.zoom / 1.16 : S.lb.zoom * 1.16;
      setLbZoom(next);
    }, { passive: false });
    // 双击在「适应 / 250%」之间切换；已放大时双击回到适应
    stage.addEventListener('dblclick', (e) => {
      if (!S.lb.open) return;
      if (e.target.closest('#lbLoading, .lb-top, .lb-foot, .lb-nav')) return;
      e.preventDefault();
      if (S.lb.zoom > 1) setLbZoom(1, true);
      else setLbZoom(2.5, true);
    });
  }

  function closeLightbox() {
    hidePhotoMenu();
    if (!S.lb.open && $('#lightbox').hidden) return;
    S.lb.open = false;
    S.lb.token++;
    $('#lightbox').hidden = true;
    const img = $('#lbImg');
    img.classList.remove('ready');
    S.lb.original = false;
    S.lb.zoom = 1;
    S.lb.pan = { x: 0, y: 0 };
    img.style.transform = '';
    $('#lbStage').classList.remove('zoomed');
    $('#lbZoomLabel').textContent = '适应';
    img.removeAttribute('src');
    $('#lbInfo').hidden = true;
    $('#lbInfoBtn').classList.remove('on');
    S.lb.info = false;
    const focus = S.lb.lastFocus;
    if (focus && focus.focus) { try { focus.focus(); } catch (_) {} }
  }

  function stepPhoto(delta) {
    hidePhotoMenu();
    if (!S.lb.open) return;
    const n = S.lb.list.length;
    if (!n) return;
    const next = clamp(S.lb.index + delta, 0, n - 1);
    if (next === S.lb.index) return;
    showPhoto(next);
  }

  async function fillInfo(id, force = false) {
    const key = String(id);
    if (force) S.detailCache.delete(key);
    let d = S.detailCache.get(key);
    if (!d) {
      $('#lbInfo').innerHTML = '<div class="loading">正在读取信息…</div>';
      try {
        d = await getJSON(`/api/photo/${encodeURIComponent(id)}`);
        S.detailCache.set(key, d);
        if (S.detailCache.size > 80) S.detailCache.delete(S.detailCache.keys().next().value);
      } catch (e) {
        if (S.lb.currentId === id) $('#lbInfo').innerHTML = `<div class="empty">读取失败：${esc(e.message)}</div>`;
        return;
      }
    }
    if (S.lb.currentId !== id || !S.lb.info) return;
    $('#lbInfo').innerHTML = infoHTML(d);
  }

  function infoHTML(d) {
    const dt = parseDate(d.taken_at);
    const people = [...new Set((d.faces || []).map((f) => f.person).filter(Boolean))];
    const rawTags = d.tags || [];
    const tags = d.tags_zh || rawTags;
    const manual = new Set(d.tags_manual || []);
    const albums = d.albums || [];
    const rows = [];
    rows.push(['时间', fmtFull(dt)]);
    if (d.width && d.height) {
      rows.push(['尺寸', `${d.width} × ${d.height} · ${(d.width * d.height / 1e6).toFixed(1)} MP`]);
    }
    if (d.device) rows.push(['设备', d.device]);
    if (d.event_title) rows.push(['事件', d.event_title]);
    if (d.city) rows.push(['地点', d.city]);
    if (people.length) rows.push(['人物', people.join('、')]);
    const body = rows.map(([k, v]) => `<div class="li-row"><span class="li-k">${esc(k)}</span><span class="li-v">${esc(v)}</span></div>`).join('');
    const tagHTML = `<div class="li-row"><span class="li-k">标签</span><span class="li-v">
      <span class="tagchips">${rawTags.map((t, i) => `<span class="tagchip ${manual.has(t) ? 'manual' : ''}">${esc(tags[i] || t)}<button data-action="tagRemove" data-tag="${esc(t)}" title="移除标签">×</button></span>`).join('')}</span>
      <div class="tag-edit"><input class="tag-input" id="tagInput" placeholder="添加标签，回车确认" autocomplete="off"><button class="tag-add" data-action="tagAdd">添加</button></div>
    </span></div>`;
    const albumHTML = `<div class="li-row"><span class="li-k">相簿</span><span class="li-v">
      ${albums.length ? `<span class="tagchips">${albums.map((a) => `<span class="tagchip">${esc(a.name)}</span>`).join('')}</span>` : ''}
      <button class="info-btn" data-action="albumPick" data-id="${esc(d.id)}">${albums.length ? '管理相簿' : '加入相簿'}</button>
    </span></div>`;
    const ocr = d.ocr_text && d.ocr_text.trim()
      ? `<div class="ocr">${esc(d.ocr_text.trim().slice(0, 1200))}</div>`
      : '';
    const path = d.path ? `<div class="li-row"><span class="li-k">文件</span><span class="li-v" style="font-size:10px;color:rgba(255,255,255,.45)">${esc(d.path)}</span></div>` : '';
    return `<h3>${esc(d.filename || '')}</h3>${body}${tagHTML}${albumHTML}${path}${ocr}`;
  }
  async function showOriginal() {
    if (!S.lb.open) return;
    const id = S.lb.currentId;
    if (!id) return;
    S.lb.original = true;
    const token = ++S.lb.token;
    $('#lbLoading').hidden = false;
    loadLbImage(originalUrl(id), token, () => {
      if (token !== S.lb.token) return;
      toast('该格式无法在查看器中直接打开，已保留高清预览', true);
      S.lb.original = false;
      showPhoto(S.lb.index);
    });
  }

  /* ───────────────────────── 弹窗 / 手动整理 ───────────────────────── */

  function openModal({ title, body, actions = [], focus = '' }) {
    hidePhotoMenu();
    S.modal.open = true;
    S.modal.actions = actions;
    S.modal.batchIds = null;
    S.modal.batchTags = null;
    S.modal.lastFocus = document.activeElement;
    $('#modalTitle').textContent = title;
    $('#modalBody').innerHTML = body;
    $('#modalActions').innerHTML = actions.map((a, i) =>
      `<button class="modal-btn ${a.cls || ''}" data-i="${i}">${esc(a.label)}</button>`).join('');
    $('#modal').hidden = false;
    setTimeout(() => {
      const el = (focus && $(focus, $('#modalBody'))) || $('#modalBody input') || $('#modalActions .primary') || $('#modalActions button');
      if (el) el.focus();
    }, 0);
  }

  function closeModal() {
    if (!S.modal.open) return;
    S.modal.open = false;
    S.modal.actions = [];
    $('#modal').hidden = true;
    const focus = S.modal.lastFocus;
    S.modal.lastFocus = null;
    if (focus && focus.focus) { try { focus.focus(); } catch (_) {} }
  }

  async function syncAlbums() {
    const d = await getJSON('/api/albums');
    S.albums = d.albums || [];
    S.viewCache.clear();
    renderAlbumNav();
    updateNav(S.current ? S.current.route : { view: 'library' });
  }

  async function createAlbumFlow(photoId) {
    const ids = Array.isArray(photoId) ? photoId.map(String) : photoId ? [String(photoId)] : [];
    const n = ids.length;
    openModal({
      title: n ? `新建相簿并加入 ${pix(n)} 张照片` : '新建相簿',
      body: `<input id="albumNameInput" maxlength="40" placeholder="相簿名称，例如：周末露营" autocomplete="off">`,
      actions: [
        { label: '取消', onClick: closeModal },
        {
          label: n ? '新建并加入' : '创建', cls: 'primary',
          onClick: async (btn) => {
            const input = $('#albumNameInput');
            const name = (input && input.value || '').trim();
            if (!name) { toast('请输入相簿名称', true); input && input.focus(); return; }
            btn.disabled = true;
            try {
              const r = await postJSON('/api/albums', { name });
              if (ids.length) {
                await postJSON(`/api/album/${r.id}/photos`, { ids });
                ids.forEach((id) => S.detailCache.delete(String(id)));
              }
              await syncAlbums();
              closeModal();
              toast(`已创建相簿「${r.name}」${ids.length ? `，加入 ${pix(ids.length)} 张` : ''}`);
              if (ids.length === 1 && S.lb.open && S.lb.info) {
                fillInfo(ids[0], true);
              } else {
                setSelectMode(false);
                S.viewCache.clear();
                go({ view: 'album', album: r.id, name: r.name }, { force: true });
              }
            } catch (e) {
              btn.disabled = false;
              toast(e.message, true);
            }
          },
        },
      ],
    });
  }

  function mergeTagsFlow() {
    const tags = [...S.tagPicked];
    if (!tags.length) { toast('先勾选要合并的标签', true); return; }
    const def = `标签合集 ${tags.length}`;
    openModal({
      title: '合并为相簿',
      body: `<input id="albumNameInput" maxlength="40" value="${esc(def)}" autocomplete="off">
        <div class="modal-hint">把选中的 ${pix(tags.length)} 个标签下的照片合并到一个新相簿：${esc(tags.join('、'))}</div>`,
      actions: [
        { label: '取消', onClick: closeModal },
        {
          label: '创建相簿', cls: 'primary',
          onClick: async (btn) => {
            const name = ($('#albumNameInput') && $('#albumNameInput').value || '').trim();
            if (!name) { toast('请输入相簿名称', true); return; }
            btn.disabled = true;
            try {
              const r = await postJSON('/api/albums/merge', { name, tags });
              S.tagPicked.clear();
              await syncAlbums();
              closeModal();
              S.viewCache.clear();
              toast(`已合并 ${pix(r.count)} 张照片到「${r.name}」`);
              go({ view: 'album', album: r.id, name: r.name }, { force: true });
            } catch (e) {
              btn.disabled = false;
              toast(e.message, true);
            }
          },
        },
      ],
    });
  }

  function openExistingAlbumPicker(tags) {
    const tagList = [...new Set((tags || []).map(String))].filter(Boolean);
    if (!tagList.length) { toast('先勾选标签', true); return; }
    const albums = S.albums.filter((a) => a.kind === 'manual');
    const list = albums.map((a) =>
      `<button class="picker-item" data-album-id="${esc(a.id)}">
        ${icon('folder')}<span>${esc(a.name)}</span><em>${pix(a.photo_count || 0)} 张</em>
      </button>`).join('');
    openModal({
      title: '把标签照片加入相簿',
      body: albums.length
        ? `<div class="picker-list">${list}</div>
           <div class="modal-hint">会把这些标签下的所有照片加入所选相簿。</div>`
        : '<div class="picker-empty">还没有手动相簿，先新建一个。</div>',
      actions: [
        { label: '新建相簿', onClick: () => { closeModal(); mergeTagsFlow(); } },
        { label: '取消', onClick: closeModal },
      ],
    });
    S.modal.batchTags = tagList;
  }

  async function addTagsToAlbum(aid, tags, btn) {
    if (btn && btn.disabled) return;
    if (btn) btn.disabled = true;
    const album = S.albums.find((a) => String(a.id) === String(aid));
    try {
      const r = await postJSON(`/api/album/${aid}/tags`, { tags });
      await syncAlbums();
      closeModal();
      S.tagPicked.clear();
      if (S.current && S.current.route.view === 'tags') updateTagBulk();
      S.viewCache.clear();
      toast(`已把 ${pix(r.count)} 张照片加入「${album ? album.name : '相簿'}」`);
    } catch (e) {
      if (btn) btn.disabled = false;
      toast(e.message, true);
    }
  }

  async function pickLocalFolder() {
    const inv = window.__TAURI__ && (window.__TAURI__.core || window.__TAURI__);
    if (inv && typeof inv.invoke === 'function') {
      try {
        return await inv.invoke('pick_folder') || '';
      } catch (_) { /* 原生选择器不可用时退回文本输入 */ }
    }
    return prompt('输入照片文件夹路径') || '';
  }

  async function importFolderFlow() {
    const dir = String(await pickLocalFolder() || '').trim().replace(/[\/]+$/, '');
    if (!dir) return;
    const base = (dir.split(/[\/]/).pop() || '导入相簿').slice(0, 40);
    openModal({
      title: '从文件夹导入相簿',
      body: `<input id="albumNameInput" maxlength="40" value="${esc(base)}" autocomplete="off">
        <div class="modal-path">${esc(dir)}</div>
        <div class="modal-hint">会扫描该文件夹及其子文件夹，已有照片直接加入；新照片在后台建立索引，完成后点刷新查看。</div>`,
      actions: [
        { label: '取消', onClick: closeModal },
        {
          label: '开始导入', cls: 'primary',
          onClick: async (btn) => {
            const name = ($('#albumNameInput') && $('#albumNameInput').value || '').trim();
            if (!name) { toast('请输入相簿名称', true); return; }
            btn.disabled = true;
            try {
              const r = await postJSON('/api/albums/import', { name, dir });
              S.importingAlbum = r.id;
              S.wasRunning = !!r.indexing;
              await syncAlbums();
              closeModal();
              S.viewCache.clear();
              toast(r.count ? `已先加入 ${pix(r.count)} 张，后台继续扫描…` : '已创建相簿，正在扫描文件夹…');
              go({ view: 'album', album: r.id, name: r.name }, { force: true });
              pollStatus();
            } catch (e) {
              btn.disabled = false;
              toast(e.message, true);
            }
          },
        },
      ],
    });
  }

  function renameAlbum(aid) {
    const a = S.albums.find((x) => String(x.id) === String(aid));
    if (!a) return;
    openModal({
      title: '重命名相簿',
      body: `<input id="albumNameInput" maxlength="40" value="${esc(a.name)}" autocomplete="off">`,
      actions: [
        { label: '取消', onClick: closeModal },
        {
          label: '保存', cls: 'primary',
          onClick: async (btn) => {
            const name = ($('#albumNameInput') && $('#albumNameInput').value || '').trim();
            if (!name) { toast('请输入相簿名称', true); return; }
            btn.disabled = true;
            try {
              const r = await postJSON(`/api/album/${aid}`, { action: 'rename', name });
              await syncAlbums();
              closeModal();
              toast('相簿已重命名');
              if (S.current && S.current.route.view === 'album' && String(S.current.route.album) === String(aid)) {
                const route = { ...S.current.route, name: r.name };
                S.viewCache.clear();
                go(route, { force: true, historyMode: 'replace' });
              }
            } catch (e) {
              btn.disabled = false;
              toast(e.message, true);
            }
          },
        },
      ],
    });
  }

  function deleteAlbum(aid) {
    const a = S.albums.find((x) => String(x.id) === String(aid));
    if (!a) return;
    openModal({
      title: '删除相簿',
      body: `确定删除相簿「${esc(a.name)}」？照片本身不会被删除。`,
      actions: [
        { label: '取消', onClick: closeModal },
        {
          label: '删除', cls: 'danger',
          onClick: async (btn) => {
            btn.disabled = true;
            try {
              await postJSON(`/api/album/${aid}`, { action: 'delete' });
              await syncAlbums();
              closeModal();
              S.viewCache.clear();
              toast(`已删除相簿「${a.name}」`);
              if (S.current && S.current.route.view === 'album' && String(S.current.route.album) === String(aid)) {
                go({ view: 'library' }, { force: true });
              }
            } catch (e) {
              btn.disabled = false;
              toast(e.message, true);
            }
          },
        },
      ],
    });
  }

  async function openAlbumPicker(pid) {
    let d = S.detailCache.get(String(pid));
    if (!d || !d.albums) {
      try {
        d = await getJSON(`/api/photo/${encodeURIComponent(pid)}`);
        S.detailCache.set(String(pid), d);
      } catch (e) {
        toast(e.message, true);
        return;
      }
    }
    const member = new Set((d.albums || []).map((a) => String(a.id)));
    const albums = S.albums.filter((a) => a.kind === 'manual');
    const list = albums.map((a) => {
      const on = member.has(String(a.id));
      return `<button class="picker-item" data-album-id="${esc(a.id)}" data-member="${on ? '1' : '0'}">
        ${icon('folder')}<span>${esc(a.name)}</span><em>${on ? '✓ 已加入' : `${pix(a.photo_count)} 张`}</em>
      </button>`;
    }).join('');
    openModal({
      title: '加入相簿',
      body: albums.length
        ? `<div class="picker-list">${list}</div>`
        : `<div class="picker-empty">还没有手动相簿，先新建一个。</div>`,
      actions: [
        { label: '新建相簿', onClick: () => { closeModal(); createAlbumFlow(pid); } },
        { label: '完成', cls: 'primary', onClick: closeModal },
      ],
    });
  }

  function openBatchAlbumPicker(ids) {
    const listIds = (ids || []).map(String);
    if (!listIds.length) return;
    const albums = S.albums.filter((a) => a.kind === 'manual');
    const list = albums.map((a) =>
      `<button class="picker-item" data-album-id="${esc(a.id)}">
        ${icon('folder')}<span>${esc(a.name)}</span><em>${pix(a.photo_count)} 张</em>
      </button>`).join('');
    openModal({
      title: `把 ${pix(listIds.length)} 张照片加入相簿`,
      body: albums.length
        ? `<div class="picker-list">${list}</div>`
        : `<div class="picker-empty">还没有手动相簿，先新建一个。</div>`,
      actions: [
        { label: '新建相簿并加入', onClick: () => { closeModal(); createAlbumFlow(listIds); } },
        { label: '取消', onClick: closeModal },
      ],
    });
    S.modal.batchIds = listIds;
  }

  async function addPhotosToAlbum(aid, ids, btn) {
    if (btn && btn.disabled) return;
    if (btn) btn.disabled = true;
    const album = S.albums.find((a) => String(a.id) === String(aid));
    try {
      await postJSON(`/api/album/${aid}/photos`, { action: 'add', ids });
      ids.forEach((id) => S.detailCache.delete(String(id)));
      await syncAlbums();
      closeModal();
      setSelectMode(false);
      toast(`已把 ${pix(ids.length)} 张照片加入「${album ? album.name : '相簿'}」`);
    } catch (e) {
      if (btn) btn.disabled = false;
      toast(e.message, true);
    }
  }

  async function toggleAlbumPhoto(aid, pid, btn) {
    if (btn && btn.disabled) return;
    if (btn) btn.disabled = true;
    const d = S.detailCache.get(String(pid)) || { albums: [] };
    const inAlbum = (d.albums || []).some((a) => String(a.id) === String(aid));
    const action = inAlbum ? 'remove' : 'add';
    const album = S.albums.find((a) => String(a.id) === String(aid));
    try {
      await postJSON(`/api/album/${aid}/photos`, { action, ids: [pid] });
      if (inAlbum) {
        d.albums = (d.albums || []).filter((a) => String(a.id) !== String(aid));
      } else {
        d.albums = [...(d.albums || []), { id: Number(aid), name: album ? album.name : '', kind: 'manual' }];
      }
      S.detailCache.set(String(pid), d);
      await syncAlbums();
      closeModal();
      if (S.lb.open && S.lb.info) fillInfo(pid, true);
      toast(inAlbum ? `已从「${album ? album.name : '相簿'}」移出` : `已加入「${album ? album.name : '相簿'}」`);
    } catch (e) {
      if (btn) btn.disabled = false;
      toast(e.message, true);
    }
  }

  async function addManualTag(pid, raw) {
    const tag = String(raw || '').trim();
    if (!tag) return;
    const input = $('#tagInput');
    if (input) input.disabled = true;
    try {
      const r = await postJSON(`/api/photo/${pid}/tags`, { action: 'add', tag });
      if (r.photo) {
        S.detailCache.set(String(pid), r.photo);
        if (S.lb.open && S.lb.info) $('#lbInfo').innerHTML = infoHTML(r.photo);
      }
      S.viewCache.clear();
      toast(`已添加标签「${r.tag || tag}」`);
      loadChrome();
    } catch (e) {
      if (input) { input.disabled = false; input.focus(); }
      toast(e.message, true);
    }
  }

  async function removeManualTag(pid, tag) {
    try {
      const r = await postJSON(`/api/photo/${pid}/tags`, { action: 'remove', tag });
      if (r.photo) {
        S.detailCache.set(String(pid), r.photo);
        if (S.lb.open && S.lb.info) $('#lbInfo').innerHTML = infoHTML(r.photo);
      }
      S.viewCache.clear();
      toast(`已移除标签「${tag}」`);
      loadChrome();
    } catch (e) {
      toast(e.message, true);
    }
  }

  function deleteTag(tag) {
    openModal({
      title: '删除标签',
      body: `确定删除标签「${esc(tag)}」？它会从所有照片移除，重建标签时也不会再出现。`,
      actions: [
        { label: '取消', onClick: closeModal },
        {
          label: '删除', cls: 'danger',
          onClick: async (btn) => {
            btn.disabled = true;
            try {
              await postJSON('/api/tags/delete', { tag });
              S.tagPicked.delete(tag);
              closeModal();
              toast(`已删除标签「${tag}」`);
              if (S.current && S.current.route.view === 'tags') {
                S.viewCache.clear();
                go({ view: 'tags' }, { force: true, historyMode: 'replace' });
              }
              loadChrome();
            } catch (e) {
              btn.disabled = false;
              toast(e.message, true);
            }
          },
        },
      ],
    });
  }

  /* ───────────────────────── 菜单 / 操作 ───────────────────────── */

  function showPopover(anchor) {
    hidePhotoMenu();
    const pop = $('#popover');
    pop.hidden = false;
    const r = anchor.getBoundingClientRect();
    const w = 218;
    pop.style.left = `${clamp(r.right - w, 8, window.innerWidth - w - 8)}px`;
    pop.style.top = `${Math.min(r.bottom + 6, window.innerHeight - 210)}px`;
  }

  function hidePopover() {
    $('#popover').hidden = true;
  }

  function showPhotoMenu(id, x, y, origin, keyboard = false) {
    const menu = $('#photoMenu');
    if (!menu) return;
    hidePopover();
    hideCtxMenu();
    S.lb.id = String(id);
    S.menu.id = String(id);
    S.menu.lbId = String(id);
    S.menu.lastFocus = origin || document.activeElement;
    menu.dataset.id = S.lb.id;
    const selected = S.select.on && S.select.ids.has(S.lb.id);
    const selectLabel = $('[data-menu-act="select"] span', menu);
    if (selectLabel) selectLabel.textContent = selected ? '取消选择' : '选择';
    menu.hidden = false;
    const pad = 8;
    const w = menu.offsetWidth || 226;
    const h = menu.offsetHeight || 200;
    menu.style.left = `${clamp(x, pad, window.innerWidth - w - pad)}px`;
    menu.style.top = `${clamp(y, pad, window.innerHeight - h - pad)}px`;
    if (keyboard) {
      const first = $('button', menu);
      if (first) first.focus({ preventScroll: true });
    }
  }

  function showPhotoInfo(id, focusTag) {
    const same = S.lb.open && String(S.lb.currentId) === String(id);
    if (!same) openLightbox(id);
    S.lb.info = true;
    $('#lbInfo').hidden = false;
    $('#lbInfoBtn').classList.add('on');
    fillInfo(S.lb.currentId).then(() => {
      if (!focusTag) return;
      const input = $('#tagInput');
      if (input) input.focus({ preventScroll: true });
    });
  }

  async function runPhotoMenu(act) {
    const id = S.menu.id;
    hidePhotoMenu();
    if (!id) return;
    const origin = S.menu.lastFocus;
    if (origin && origin.isConnected && origin.focus) {
      try { origin.focus({ preventScroll: true }); } catch (_) {}
    }
    if (act === 'view') {
      if (!(S.lb.open && String(S.lb.currentId) === String(id))) openLightbox(id);
      return;
    }
    if (act === 'original') {
      if (!(S.lb.open && String(S.lb.currentId) === String(id))) openLightbox(id);
      showOriginal();
      return;
    }
    if (act === 'album') { await openAlbumPicker(id); return; }
    if (act === 'info' || act === 'tags') { showPhotoInfo(id, act === 'tags'); return; }
    if (act === 'select') {
      if (S.lb.open) closeLightbox();
      if (!S.select.on) setSelectMode(true);
      togglePhotoSelect(id, false);
    }
  }

  // 单一定义：隐藏照片菜单时同步清空状态与通用右键菜单，避免两个菜单叠加/残留。
  function hidePhotoMenu() {
    const menu = $('#photoMenu');
    if (menu) { menu.hidden = true; menu.dataset.id = ''; }
    S.menu.id = null;
    S.menu.lbId = null;
    S.lb.id = null;
    hideCtxMenu();
  }

  function hideCtxMenu() {
    const menu = $('#ctxMenu');
    if (!menu) return;
    menu.hidden = true;
    menu.innerHTML = '';
    S.menu.ctx = null;
    S.menu.ctxFocus = null;
  }

  function showCtxMenu(items, x, y, origin, keyboard = false) {
    const menu = $('#ctxMenu');
    if (!menu || !items || !items.length) return;
    hidePopover();
    hidePhotoMenu();  // 同时隐藏照片菜单与旧的通用菜单，避免两个菜单叠加
    menu.innerHTML = items.map((it, i) => {
      if (it.sep) return '<div class="menu-sep"></div>';
      const ic = it.icon ? icon(it.icon) : '';
      const kb = it.key ? `<kbd>${esc(it.key)}</kbd>` : '';
      return `<button data-ctx-i="${i}" ${it.danger ? 'class="danger"' : ''} ${it.disabled ? 'disabled' : ''}>${ic}<span>${esc(it.label)}</span>${kb}</button>`;
    }).join('');
    S.menu.ctx = items;
    S.menu.ctxFocus = origin || document.activeElement;
    menu.hidden = false;
    const pad = 8;
    const w = menu.offsetWidth || 232;
    const h = menu.offsetHeight || 180;
    menu.style.left = `${clamp(x, pad, window.innerWidth - w - pad)}px`;
    menu.style.top = `${clamp(y, pad, window.innerHeight - h - pad)}px`;
    menu.dataset.keep = '1';
    requestAnimationFrame(() => { if (menu) delete menu.dataset.keep; });
    if (keyboard) {
      const first = $('button:not([disabled])', menu);
      if (first) first.focus({ preventScroll: true });
    }
  }

  function runCtxMenu(i) {
    const items = S.menu.ctx || [];
    const focus = S.menu.ctxFocus;
    hideCtxMenu();
    const act = items[i] && items[i].action;
    if (!act) return;
    if (focus && focus.isConnected && focus.focus) {
      try { focus.focus({ preventScroll: true }); } catch (_) {}
    }
    try {
      const r = act();
      if (r && typeof r.catch === 'function') r.catch((err) => toast(err.message || '操作失败', true));
    } catch (err) {
      toast(err.message || '操作失败', true);
    }
  }

  function albumCtxItems(aid) {
    const a = S.albums.find((x) => String(x.id) === String(aid));
    const id = a ? a.id : aid;
    return [
      { label: '打开相簿', icon: 'folder', action: () => go({ view: 'album', album: id, name: a ? a.name : '' }) },
      {
        label: '添加图片文件夹', icon: 'plus', hidden: !a || a.kind !== 'manual',
        action: async () => {
          const dir = await pickLocalFolder();
          if (!dir) return;
          const r = await postJSON(`/api/album/${id}/import`, { dir });
          S.importingAlbum = r.id;
          S.wasRunning = !!r.indexing;
          await syncAlbums();
          S.viewCache.clear();
          toast(r.count ? `已先加入 ${pix(r.count)} 张，后台继续扫描…` : '已创建相簿，正在扫描文件夹…');
          go({ view: 'album', album: r.id, name: r.name }, { force: true });
          pollStatus();
        },
      },
      { sep: true },
      { label: '重命名', icon: 'tag', action: () => renameAlbum(id) },
      { label: '删除相簿', icon: 'close', danger: true, action: () => deleteAlbum(id) },
    ].filter((x) => !x.hidden);
  }

  function albumAreaCtxItems(aid) {
    const a = S.albums.find((x) => String(x.id) === String(aid));
    return [
      { label: '添加图片文件夹', icon: 'plus', action: albumCtxItems(aid).find((x) => x.label === '添加图片文件夹').action },
      { label: '新建相簿', icon: 'folder', action: () => createAlbumFlow() },
      { sep: true },
      { label: '重命名', icon: 'tag', action: () => renameAlbum(aid) },
      { label: '删除相簿', icon: 'close', danger: true, action: () => deleteAlbum(aid) },
    ];
  }

  function personCtxItems(pid) {
    const p = personById(pid);
    return [
      { label: p ? `打开「${p.name || '未命名'}」` : '打开人物', icon: 'people', action: () => go({ view: 'person', person: pid, name: p ? p.name : '' }) },
      { label: '重命名', icon: 'tag', action: () => renamePersonFlow(pid, p) },
    ];
  }

  function personById(pid) {
    const st = S.current;
    if (st && st.kind === 'people') return (st.items || []).find((p) => String(p.id) === String(pid));
    if (st && st.route && st.route.view === 'person' && String(st.route.person) === String(pid)) {
      return { id: st.route.person, name: st.route.name || '' };
    }
    return null;
  }

  function renamePersonFlow(pid, person) {
    const name = (person && person.name) || '';
    openModal({
      title: '重命名人物',
      body: `<input id="personNameInput" maxlength="40" value="${esc(name)}" autocomplete="off" placeholder="例如：小明">
        <div class="modal-hint">名字会应用到这个人物的所有识别脸。</div>`,
      actions: [
        { label: '取消', onClick: closeModal },
        {
          label: '保存', cls: 'primary',
          onClick: async (btn) => {
            const input = $('#personNameInput');
            const value = (input && input.value || '').trim();
            if (!value) { toast('请输入人物名字', true); input && input.focus(); return; }
            btn.disabled = true;
            try {
              const r = await postJSON(`/api/person/${Number(pid)}`, { name: value });
              await syncAlbums();
              closeModal();
              S.viewCache.clear();
              if (S.current && S.current.kind === 'people') {
                const p = S.current.items.find((x) => String(x.id) === String(pid));
                if (p) p.name = r.name;
              }
              toast(`已命名人物「${r.name}」`);
              loadChrome();
              const route = S.current && S.current.route;
              if (route && route.view === 'person' && String(route.person) === String(pid)) {
                go({ ...route, name: r.name }, { force: true, historyMode: 'replace' });
              }
            } catch (e) {
              btn.disabled = false;
              toast(e.message, true);
            }
          },
        },
      ],
    });
  }

  async function refreshAll() {
    const route = S.current ? S.current.route : { view: 'library' };
    if (S.current) {
      S.viewCache.delete(S.current.key);
      S.current = null;
    }
    S.detailCache.clear();
    S.lastRefresh = Date.now();
    loadChrome();
    await go(route, { force: true, historyMode: 'none' });
    updateFooter();
  }

  async function startIndex() {
    const dir = String(await pickLocalFolder() || '').trim().replace(/[\/]+$/, '');
    if (!dir) return;
    try {
      const endpoint = S.stats && S.stats.photos ? '/api/index/start' : '/api/setup';
      const r = await postJSON(endpoint, { dir });
      S.libraryDir = r.dir || dir;
      toast(`正在添加照片：${S.libraryDir}`);
      S.wasRunning = true;
      pollStatus();
    } catch (e) {
      toast(e.message, true);
    }
  }

  async function startRetag() {
    if (!confirm('重建智能标签？会用本地向量重新打标，不重新解码图片，通常几秒到几十秒。')) return;
    try {
      await postJSON('/api/retag/start', {});
      toast('开始重建智能标签');
      S.wasRunning = true;
      pollStatus();
    } catch (e) {
      toast(e.message, true);
    }
  }

  function resolvedTheme() {
    if (S.theme !== 'auto') return S.theme;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  function toggleTheme() {
    S.theme = resolvedTheme() === 'dark' ? 'light' : 'dark';
    localStorage.setItem('garden.theme', S.theme);
    applyTheme();
  }

  function useSystemTheme() {
    S.theme = 'auto';
    localStorage.setItem('garden.theme', S.theme);
    applyTheme();
    toast('外观已跟随系统');
  }

  function applyTheme() {
    let resolved = S.theme;
    if (S.theme === 'auto') {
      resolved = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    document.documentElement.dataset.theme = resolved;
    const btn = $('#btnTheme');
    if (btn) {
      const name = S.theme === 'auto' ? 'monitor' : S.theme === 'light' ? 'sun' : 'moon';
      btn.innerHTML = icon(name);
      btn.title = S.theme === 'auto' ? '外观：跟随系统' : S.theme === 'light' ? '外观：浅色' : '外观：深色';
    }
  }

  function updateSegState() {
    const seg = $('#segLayout');
    const view = S.current && S.current.route.view;
    const gridLike = ['library', 'event', 'person', 'album', 'city', 'shots', 'tag'].includes(view);
    const selectable = [...['library', 'event', 'person', 'album', 'city', 'shots', 'tag'], 'search'].includes(view);
    seg.classList.toggle('disabled', !gridLike);
    $$('#segLayout button').forEach((b) => {
      const on = b.dataset.layout === S.layout;
      b.classList.toggle('on', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    const sel = $('#btnSelect');
    if (sel) {
      sel.hidden = !selectable;
      sel.classList.toggle('on', S.select.on);
      sel.setAttribute('aria-pressed', S.select.on ? 'true' : 'false');
      const label = $('span', sel);
      if (label) label.textContent = S.select.on ? '完成' : '选择';
    }
    if (!selectable && S.select.on) setSelectMode(false);
    updateBatchUI();
  }

  function updateBatchUI() {
    const bar = $('#batchBar');
    if (!bar) return;
    const n = S.select.ids.size;
    bar.hidden = !S.select.on;
    const count = $('#batchCount');
    if (count) count.textContent = `已选 ${pix(n)} 张`;
    const add = $('#batchAddAlbum');
    if (add) add.disabled = !n;
    document.body.classList.toggle('select-mode', S.select.on);
  }

  function setSelectMode(on) {
    hidePhotoMenu();
    S.select.on = !!on;
    if (!on) {
      S.select.ids.clear();
      S.select.last = null;
    }
    if (S.current) paintSelectionTiles(S.current);
    updateSegState();
  }

  function paintSelectionTiles(st) {
    $$('.tile', st.node).forEach((tile) => {
      const picked = S.select.on && S.select.ids.has(String(tile.dataset.id));
      tile.classList.toggle('selected', picked);
      tile.setAttribute('aria-selected', picked ? 'true' : 'false');
      const chk = $('.tile-check', tile);
      if (chk) chk.innerHTML = picked ? icon('check') : '';
    });
  }

  function togglePhotoSelect(id, extend) {
    if (!S.select.on || !S.current) return;
    const sid = String(id);
    const st = S.current;
    if (extend && S.select.last && st.photoList.includes(Number(S.select.last))) {
      const a = st.photoList.indexOf(Number(S.select.last));
      const b = st.photoList.indexOf(Number(id));
      if (a >= 0 && b >= 0) {
        const lo = Math.min(a, b);
        const hi = Math.max(a, b);
        const target = !S.select.ids.has(sid);
        for (let i = lo; i <= hi; i++) {
          const x = String(st.photoList[i]);
          if (target) S.select.ids.add(x); else S.select.ids.delete(x);
        }
      }
    } else if (S.select.ids.has(sid)) {
      S.select.ids.delete(sid);
    } else {
      S.select.ids.add(sid);
    }
    S.select.last = sid;
    paintSelectionTiles(st);
    updateSegState();
  }

  function setLayout(layout) {
    if (!['all', 'year', 'month', 'day'].includes(layout)) return;
    S.layout = layout;
    localStorage.setItem('garden.layout', layout);
    updateSegState();
    const st = S.current;
    if (!st) return;
    if (st.kind === 'grid') {
      const y = scroller.scrollTop;
      renderGrid(st);
      scroller.scrollTop = y;
    }
  }

  function setZoom(v) {
    S.zoom = clamp(+v || 170, 110, 280);
    localStorage.setItem('garden.zoom', String(S.zoom));
    document.documentElement.style.setProperty('--tile', `${S.zoom}px`);
    const input = $('#zoom');
    if (input && +input.value !== S.zoom) input.value = String(S.zoom);
  }

  function openNav(nav) {
    const routes = {
      library: { view: 'library' },
      memories: { view: 'memories' },
      people: { view: 'people' },
      places: { view: 'places' },
      clean: { view: 'clean' },
      dup: { view: 'dup' },
      shots: { view: 'shots' },
      text: { view: 'text' },
      bursts: { view: 'bursts' },
      tags: { view: 'tags' },
    };
    if (routes[nav]) go(routes[nav]);
  }

  /* ───────────────────────── 事件绑定 ───────────────────────── */

  function bind() {
    $('#sidebar').addEventListener('click', (e) => {
      const newAlbum = e.target.closest('[data-action="albumNew"]');
      if (newAlbum) { createAlbumFlow(); return; }
      const importFolder = e.target.closest('[data-action="albumImport"]');
      if (importFolder) { importFolderFlow(); return; }
      const album = e.target.closest('.album-item');
      if (album) {
        go({ view: 'album', album: album.dataset.album, name: album.dataset.name });
        return;
      }
      const item = e.target.closest('.nav-item[data-view]');
      if (item) openNav(item.dataset.view);
    });

    $('#sidebar').addEventListener('contextmenu', (e) => {
      const album = e.target.closest('.album-item');
      if (!album) return;
      e.preventDefault();
      const a = S.albums.find((x) => String(x.id) === String(album.dataset.album));
      showCtxMenu(
        a && a.kind === 'manual' ? albumCtxItems(album.dataset.album) : [
          { label: '打开相簿', icon: 'folder', action: () => go({ view: 'album', album: album.dataset.album, name: album.dataset.name }) },
        ],
        e.clientX, e.clientY, album,
      );
    });

    // 侧栏相簿的键盘右键（Shift+F10 / 菜单键），与内容区保持一致
    $('#sidebar').addEventListener('keydown', (e) => {
      if (e.key !== 'ContextMenu' && !(e.shiftKey && e.key === 'F10')) return;
      const album = e.target.closest('.album-item');
      if (!album) return;
      e.preventDefault();
      const r = album.getBoundingClientRect();
      const a = S.albums.find((x) => String(x.id) === String(album.dataset.album));
      showCtxMenu(
        a && a.kind === 'manual' ? albumCtxItems(album.dataset.album) : [
          { label: '打开相簿', icon: 'folder', action: () => go({ view: 'album', album: album.dataset.album, name: album.dataset.name }) },
        ],
        r.left + 24, r.top + 24, album, true,
      );
    });

    scroller.addEventListener('click', (e) => {
      const nav = e.target.closest('[data-action="nav"]');
      if (nav) { openNav(nav.dataset.nav); return; }

      const clear = e.target.closest('[data-action="clearSearch"]');
      if (clear) { go({ view: 'library' }); return; }

      const goLib = e.target.closest('[data-action="goLibrary"]');
      if (goLib) { go({ view: 'library' }); return; }

      const more = e.target.closest('[data-action="more"]');
      if (more && S.current) { loadMore(S.current); return; }

      const retry = e.target.closest('[data-action="retry"]');
      if (retry) { refreshAll(); return; }

      const scan = e.target.closest('[data-action="scan"]');
      if (scan) { startIndex(); return; }

      const tagDelete = e.target.closest('[data-action="tagDelete"]');
      if (tagDelete) { deleteTag(tagDelete.dataset.tag); return; }

      const tagCheck = e.target.closest('.tag-check');
      if (tagCheck) {
        toggleTagPick(tagCheck.dataset.tag, !S.tagPicked.has(tagCheck.dataset.tag));
        return;
      }

      const tagAll = e.target.closest('[data-action="tagAll"]');
      if (tagAll) {
        $$('.tag-admin', S.current.node).filter((el) => !el.hidden).forEach((el) => S.tagPicked.add(el.dataset.tag));
        updateTagBulk();
        return;
      }
      const tagClear = e.target.closest('[data-action="tagClear"]');
      if (tagClear) { S.tagPicked.clear(); updateTagBulk(); return; }
      const tagMerge = e.target.closest('[data-action="tagMerge"]');
      if (tagMerge) { mergeTagsFlow(); return; }
      const tagToAlbum = e.target.closest('[data-action="tagToAlbum"]');
      if (tagToAlbum) { openExistingAlbumPicker([...S.tagPicked]); return; }

      const tagView = e.target.closest('[data-action="tagView"]');
      if (tagView) {
        go({ view: 'tag', tag: tagView.dataset.tag, name: tagView.dataset.name || tagView.dataset.tag });
        return;
      }

      const rename = e.target.closest('[data-action="renameAlbum"]');
      if (rename) { renameAlbum(rename.dataset.id); return; }

      const delAlbum = e.target.closest('[data-action="deleteAlbum"]');
      if (delAlbum) { deleteAlbum(delAlbum.dataset.id); return; }

      const renamePerson = e.target.closest('[data-action="renamePerson"]');
      if (renamePerson) { renamePersonFlow(renamePerson.dataset.id, personById(renamePerson.dataset.id)); return; }

      const ph = e.target.closest('[data-action="photo"]');
      if (ph) {
        if (S.select.on) togglePhotoSelect(ph.dataset.id, e.shiftKey);
        else openLightbox(ph.dataset.id);
        return;
      }

      const ev = e.target.closest('[data-action="event"]');
      if (ev) {
        go({ view: 'event', event: ev.dataset.id, name: ev.dataset.name, city: ev.dataset.city });
        return;
      }

      const person = e.target.closest('[data-action="person"]');
      if (person) {
        go({ view: 'person', person: person.dataset.id, name: person.dataset.name });
        return;
      }
    });

    scroller.addEventListener('contextmenu', (e) => {
      const tile = e.target.closest('[data-action="photo"]');
      if (tile) {
        e.preventDefault();
        showPhotoMenu(tile.dataset.id, e.clientX, e.clientY, tile);
        return;
      }
      const personCard = e.target.closest('.person');
      if (personCard) {
        e.preventDefault();
        showCtxMenu(personCtxItems(personCard.dataset.id), e.clientX, e.clientY, personCard);
        return;
      }
      let aid = null;
      if (S.current && S.current.route.view === 'album') {
        aid = S.current.route.album;
      } else {
        const albumCard = e.target.closest('.album-item');
        if (albumCard) aid = albumCard.dataset.album;
      }
      if (aid && S.albums.some((a) => String(a.id) === String(aid))) {
        e.preventDefault();
        const a = S.albums.find((x) => String(x.id) === String(aid));
        const manual = a && a.kind === 'manual';
        const items = manual
          ? albumAreaCtxItems(aid)
          : [
            { label: '打开相簿', icon: 'folder', action: () => go({ view: 'album', album: aid, name: a ? a.name : '' }) },
          ];
        showCtxMenu(items, e.clientX, e.clientY, e.target.closest('.album-item, .empty, .hero, section, .body') || e.target);
        return;
      }
    });

    $('#ctxMenu').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-ctx-i]');
      if (!btn || btn.disabled) return;
      runCtxMenu(+btn.dataset.ctxI);
    });

    $('#ctxMenu').addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { e.preventDefault(); hideCtxMenu(); }
      const btns = [...$('#ctxMenu').querySelectorAll('button:not([disabled])')];
      const i = btns.indexOf(e.target);
      if (i >= 0 && e.key === 'ArrowDown') {
        e.preventDefault();
        btns[(i + 1) % btns.length].focus({ preventScroll: true });
      } else if (i >= 0 && e.key === 'ArrowUp') {
        e.preventDefault();
        btns[(i - 1 + btns.length) % btns.length].focus({ preventScroll: true });
      }
    });

    scroller.addEventListener('scroll', hidePhotoMenu, { passive: true });

    scroller.addEventListener('input', (e) => {
      if (e.target && e.target.id === 'tagFilter') filterTags(e.target);
    });

    scroller.addEventListener('keydown', (e) => {
      if (e.key === 'ContextMenu' || (e.shiftKey && e.key === 'F10')) {
        const tile = e.target.closest('[data-action="photo"]');
        if (tile) {
          e.preventDefault();
          const r = tile.getBoundingClientRect();
          showPhotoMenu(tile.dataset.id, r.left + 24, r.top + 24, tile, true);
          return;
        }
        const personCard = e.target.closest('[data-action="person"]');
        if (personCard) {
          e.preventDefault();
          const r = personCard.getBoundingClientRect();
          showCtxMenu(personCtxItems(personCard.dataset.id), r.left + 24, r.top + 24, personCard, true);
          return;
        }
        const albumCard = e.target.closest('.album-item');
        if (albumCard) {
          e.preventDefault();
          const r = albumCard.getBoundingClientRect();
          const a = S.albums.find((x) => String(x.id) === String(albumCard.dataset.album));
          showCtxMenu(
            a && a.kind === 'manual' ? albumCtxItems(albumCard.dataset.album) : [
              { label: '打开相簿', icon: 'folder', action: () => go({ view: 'album', album: albumCard.dataset.album, name: albumCard.dataset.name }) },
            ],
            r.left + 24, r.top + 24, albumCard, true,
          );
        }
      }
      if (e.key !== 'Enter' && e.key !== ' ') return;
      // 原生 button / 复选 chip 已自带键盘激活，别再冒泡触发外层行操作
      if (e.target.closest('button, [role="checkbox"], input, textarea, select')) return;
      const el = e.target.closest('[data-action]');
      if (!el) return;
      e.preventDefault();
      el.click();
    });

    $$('#segLayout button').forEach((b) => {
      b.addEventListener('click', () => setLayout(b.dataset.layout));
    });

    $('#zoom').addEventListener('input', (e) => setZoom(e.target.value));
    $('#btnSelect').addEventListener('click', () => setSelectMode(!S.select.on));
    $('#batchAddAlbum').addEventListener('click', () => openBatchAlbumPicker([...S.select.ids]));
    $('#batchCancel').addEventListener('click', () => setSelectMode(false));
    $('#btnRefresh').addEventListener('click', () => refreshAll());
    $('#btnBack').addEventListener('click', () => {
      if (S.current && S.current.route.view === 'tag') go({ view: 'tags' });
      else go({ view: 'library' });
    });
    $('#btnSearch').addEventListener('click', () => $('#q').focus());

    $('#btnMenu').addEventListener('click', (e) => {
      e.stopPropagation();
      const pop = $('#popover');
      if (pop.hidden) showPopover(e.currentTarget); else hidePopover();
    });

    $('#popover').addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-act]');
      if (!btn) return;
      hidePopover();
      const act = btn.dataset.act;
      if (act === 'refresh') refreshAll();
      else if (act === 'retag') startRetag();
      else if (act === 'index') startIndex();
      else if (act === 'theme') useSystemTheme();
    });

    $('#photoMenu').addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-menu-act]');
      if (btn) runPhotoMenu(btn.dataset.menuAct);
    });

    $('#btnTheme').addEventListener('click', toggleTheme);
    $('#welcomePick').addEventListener('click', chooseLibrary);
    $('#welcomeContinue').addEventListener('click', continueInBackground);

    document.addEventListener('click', (e) => {
      const pop = $('#popover');
      if (!pop.hidden && !pop.contains(e.target) && !$('#btnMenu').contains(e.target)) hidePopover();
      const menu = $('#photoMenu');
      if (!menu.hidden && !menu.contains(e.target)) hidePhotoMenu();
      const ctxm = $('#ctxMenu');
      if (!ctxm.hidden && !ctxm.contains(e.target)) hideCtxMenu();
    });

    document.addEventListener('contextmenu', (e) => {
      if (e.target.closest('#photoMenu')) return;
      if (e.target.closest('#ctxMenu')) return;
      if (e.target.closest('[data-action="photo"]') || e.target.closest('#lbStage') || e.target.closest('#lbImg')) return;
      if ($('#ctxMenu').dataset.keep === '1') return;
      hidePhotoMenu();
    });

    window.addEventListener('resize', hidePhotoMenu);
    window.addEventListener('blur', hidePhotoMenu);

    const q = $('#q');
    q.addEventListener('compositionstart', () => { S.typing = true; });
    q.addEventListener('compositionend', () => {
      S.typing = false;
      scheduleSearch(true);
    });
    q.addEventListener('input', () => { if (!S.typing) scheduleSearch(false); });
    q.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const v = q.value.trim();
        if (v) go({ view: 'search', q: v });
      } else if (e.key === 'Escape') {
        q.blur();
        if (q.value) {
          q.value = '';
          go({ view: 'library' });
        }
      }
    });

    document.addEventListener('keydown', (e) => {
      const tag = (e.target && e.target.tagName) || '';
      const typing = tag === 'INPUT' || tag === 'TEXTAREA' || (e.target && e.target.isContentEditable);

      if (!$('#photoMenu').hidden) {
        if (e.key === 'Escape') { e.preventDefault(); hidePhotoMenu(); return; }
      }

      if (S.onboarding) return;

      if (S.modal.open) {
        if (e.key === 'Escape') { e.preventDefault(); closeModal(); }
        return;
      }

      if (S.lb.open) {
        if (typing) return;
        if (e.key === 'Escape') { e.preventDefault(); closeLightbox(); }
        else if (e.key === 'ArrowLeft') { e.preventDefault(); stepPhoto(-1); }
        else if (e.key === 'ArrowRight') { e.preventDefault(); stepPhoto(1); }
        else if ((e.key === 'i' || e.key === 'I') && !typing) { e.preventDefault(); toggleInfo(); }
        else if (e.key === '+' || e.key === '=') { e.preventDefault(); changeLbZoom(1.25); }
        else if (e.key === '-') { e.preventDefault(); changeLbZoom(1 / 1.25); }
        else if (e.key === '0') { e.preventDefault(); setLbZoom(1, true); }
        return;
      }

      if (S.select.on && e.key === 'Escape') {
        e.preventDefault();
        setSelectMode(false);
        return;
      }

      if ((e.metaKey || e.ctrlKey) && (e.key === 'k' || e.key === 'K')) {
        e.preventDefault();
        q.focus();
        q.select();
        return;
      }
      if ((e.metaKey || e.ctrlKey) && (e.key === 'r' || e.key === 'R')) {
        e.preventDefault();
        refreshAll();
        return;
      }
      if (e.key === 'Escape') {
        if (!typing && S.current && S.current.route.view === 'search') go({ view: 'library' });
        hidePopover();
        return;
      }
      if (e.key === '/' && !typing) {
        e.preventDefault();
        q.focus();
      }
    });

    window.addEventListener('popstate', () => {
      const r = parseHash();
      if (r) go(r, { historyMode: 'none' });
    });

    if (window.matchMedia) {
      const mq = window.matchMedia('(prefers-color-scheme: dark)');
      const onChange = () => { if (S.theme === 'auto') applyTheme(); };
      if (mq.addEventListener) mq.addEventListener('change', onChange);
      else if (mq.addListener) mq.addListener(onChange);
    }

    $('#lbClose').addEventListener('click', closeLightbox);
    $('#lbPrev').addEventListener('click', () => stepPhoto(-1));
    $('#lbNext').addEventListener('click', () => stepPhoto(1));
    $('#lbZoomIn').addEventListener('click', () => changeLbZoom(1.25));
    $('#lbZoomOut').addEventListener('click', () => changeLbZoom(1 / 1.25));
    $('#lbFitBtn').addEventListener('click', () => setLbZoom(1, true));
    bindLbPan();
    $('#lbAlbumBtn').addEventListener('click', () => {
      if (S.lb.currentId) openAlbumPicker(S.lb.currentId);
    });
    $('#lbInfoBtn').addEventListener('click', toggleInfo);
    $('#lbOpenBtn').addEventListener('click', showOriginal);
    $('#lbImg').addEventListener('click', (e) => e.stopPropagation());

    $('#lbInfo').addEventListener('click', (e) => {
      const add = e.target.closest('[data-action="tagAdd"]');
      if (add) { addManualTag(S.lb.currentId, $('#tagInput') && $('#tagInput').value); return; }
      const rm = e.target.closest('[data-action="tagRemove"]');
      if (rm) { removeManualTag(S.lb.currentId, rm.dataset.tag); return; }
      const pick = e.target.closest('[data-action="albumPick"]');
      if (pick) { openAlbumPicker(pick.dataset.id || S.lb.currentId); return; }
    });

    $('#lbInfo').addEventListener('keydown', (e) => {
      e.stopPropagation();
      if (e.key === 'Enter' && e.target && e.target.id === 'tagInput') {
        e.preventDefault();
        addManualTag(S.lb.currentId, e.target.value);
      } else if (e.key === 'Escape' && e.target && e.target.id === 'tagInput') {
        e.preventDefault();
        e.target.blur();
      }
    });

    $('#modalActions').addEventListener('click', (e) => {
      const btn = e.target.closest('[data-i]');
      if (!btn) return;
      const action = S.modal.actions[+btn.dataset.i];
      if (!action || btn.disabled) return;
      try {
        const r = action.onClick(btn);
        if (r && typeof r.catch === 'function') r.catch((err) => toast(err.message || '操作失败', true));
      } catch (err) {
        toast(err.message || '操作失败', true);
      }
    });

    $('#modalBody').addEventListener('click', (e) => {
      const item = e.target.closest('.picker-item[data-album-id]');
      if (!item) return;
      if (S.modal.batchIds) addPhotosToAlbum(item.dataset.albumId, S.modal.batchIds, item);
      else if (S.modal.batchTags) addTagsToAlbum(item.dataset.albumId, S.modal.batchTags, item);
      else toggleAlbumPhoto(item.dataset.albumId, S.lb.currentId, item);
    });

    $('#modal').addEventListener('click', (e) => {
      if (e.target === $('#modal')) closeModal();
    });

    $('#modal').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.target && e.target.tagName === 'INPUT') {
        const primary = $('#modalActions .primary');
        if (primary) { e.preventDefault(); primary.click(); }
      }
    });

    $('#lightbox').addEventListener('click', (e) => {
      const stage = $('#lbStage');
      if (e.target === $('#lightbox') || e.target === stage) closeLightbox();
    });

    $('#lightbox').addEventListener('contextmenu', (e) => {
      if (!S.lb.currentId) return;
      if (e.target.closest('#lbInfo, .lb-top, .lb-foot, .lb-nav')) return;
      e.preventDefault();
      showPhotoMenu(S.lb.currentId, e.clientX, e.clientY, $('#lbClose'));
    });

    document.addEventListener('error', (e) => {
      const img = e.target;
      if (!img || img.tagName !== 'IMG') return;
      img.classList.add('gone');
      const host = img.closest('.tile, .row, .person, .memory, .place-card');
      if (host) host.classList.add('img-broken');
    }, true);
  }

  function toggleInfo() {
    if (!S.lb.open) return;
    S.lb.info = !S.lb.info;
    $('#lbInfo').hidden = !S.lb.info;
    $('#lbInfoBtn').classList.toggle('on', S.lb.info);
    if (S.lb.info && S.lb.currentId) fillInfo(S.lb.currentId);
  }

  let searchTimer = null;
  function scheduleSearch(immediate) {
    clearTimeout(searchTimer);
    const v = $('#q').value.trim();
    if (!v) {
      if (S.current && S.current.route.view === 'search') go({ view: 'library' });
      $('#searchNote').textContent = '';
      return;
    }
    searchTimer = setTimeout(() => {
      if ($('#q').value.trim() !== v) return;
      go({ view: 'search', q: v }, { historyMode: 'replace' });
    }, immediate ? 0 : 480);
  }

  async function loadMore(st) {
    if (st.loadingMore || st.exhausted) return;
    st.loadingMore = true;
    const btn = st.node.querySelector('[data-action="more"]');
    if (btn) { btn.disabled = true; btn.textContent = '正在载入…'; }
    try {
      const params = gridParams(st.route);
      params.offset = st.photos.length;
      const data = await getJSON('/api/photos', params);
      if (S.current !== st) return;
      const more = data.photos || [];
      st.photos = st.photos.concat(more);
      st.total = data.total || st.total;
      st.exhausted = !more.length || st.photos.length >= st.total;
      st.photoList = st.photos.map((p) => p.id);
      const y = scroller.scrollTop;
      renderGrid(st);
      scroller.scrollTop = y;
    } catch (e) {
      toast(e.message, true);
    } finally {
      st.loadingMore = false;
    }
  }

  /* ───────────────────────── 首次启动 ───────────────────────── */

  function setupError(message) {
    const status = $('#welcomeStatus');
    status.textContent = message;
    status.classList.add('err');
    const btn = $('#welcomePick');
    btn.disabled = false;
    const label = $('span', btn);
    if (label) label.textContent = '重新选择';
    $('#welcomeProgress').hidden = true;
  }

  function showWelcome(message) {
    S.onboarding = true;
    S.setupWaits = 0;
    $('#app').hidden = true;
    $('#welcome').hidden = false;
    $('#welcomePick').disabled = false;
    const label = $('span', $('#welcomePick'));
    if (label) label.textContent = '选择文件夹';
    $('#welcomeProgress').hidden = true;
    $('#welcomeContinue').hidden = true;
    $('#welcomePath').hidden = true;
    $('#welcomePath').textContent = '';
    const status = $('#welcomeStatus');
    status.textContent = message || '';
    status.classList.toggle('err', !!message);
  }

  async function pollSetup() {
    if (!S.onboarding) return;
    let st;
    try {
      st = await getJSON('/api/index/status');
    } catch (_) {
      $('#welcomeStatus').textContent = '正在连接本地照片服务…';
      $('#welcomeStatus').classList.add('err');
      setTimeout(pollSetup, 900);
      return;
    }
    if (!st.running && !st.error && !st.started_at) {
      S.setupWaits = (S.setupWaits || 0) + 1;
      if (S.setupWaits > 20) {
        setupError('无法启动索引，请重新选择文件夹。');
        return;
      }
      setTimeout(pollSetup, 300);
      return;
    }
    S.setupWaits = 0;
    const done = st.done || 0;
    const total = st.total || 0;
    const pct = total ? Math.max(2, Math.min(100, Math.round((done / total) * 100))) : 4;
    $('#welcomeBar').style.width = `${pct}%`;
    if (st.running) {
      $('#welcomeStatus').classList.remove('err');
      $('#welcomeStatus').textContent = total
        ? `正在建立索引 ${pix(done)} / ${pix(total)} 张照片`
        : '正在准备本地模型并读取文件夹…';
      setTimeout(pollSetup, 500);
      return;
    }
    if (st.error) {
      setupError(`建立索引失败：${st.error}`);
      return;
    }
    let setup;
    try {
      setup = await getJSON('/api/setup');
    } catch (e) {
      setupError(`无法读取照片库状态：${e.message}`);
      return;
    }
    if (!setup.photos) {
      setupError('这个文件夹里没有找到支持的照片，请换一个文件夹。');
      return;
    }
    S.libraryDir = setup.library || S.libraryDir;
    $('#welcomeBar').style.width = '100%';
    $('#welcomeStatus').classList.remove('err');
    $('#welcomeStatus').textContent = `已建立 ${pix(setup.photos)} 张照片的索引`;
    S.onboarding = false;
    $('#welcome').hidden = true;
    $('#app').hidden = false;
    await enterApp();
  }

  async function chooseLibrary() {
    const dir = String(await pickLocalFolder() || '').trim().replace(/[\/]+$/, '');
    if (!dir) return;
    S.libraryDir = dir;
    const btn = $('#welcomePick');
    btn.disabled = true;
    const label = $('span', btn);
    if (label) label.textContent = '正在建立索引…';
    $('#welcomePath').textContent = dir;
    $('#welcomePath').hidden = false;
    $('#welcomeProgress').hidden = false;
    $('#welcomeBar').style.width = '4%';
    S.setupWaits = 0;
    const status = $('#welcomeStatus');
    status.classList.remove('err');
    status.textContent = '正在读取文件夹…';
    try {
      await postJSON('/api/setup', { dir });
      $('#welcomeContinue').hidden = false;
      await pollSetup();
    } catch (e) {
      setupError(e.message);
    }
  }

  async function continueInBackground() {
    try { S.status = await getJSON('/api/index/status'); } catch (_) {}
    S.onboarding = false;
    $('#welcome').hidden = true;
    $('#app').hidden = false;
    if (S.status && S.status.running) S.wasRunning = true;
    await enterApp();
  }

  async function enterApp() {
    $('#welcome').hidden = true;
    $('#app').hidden = false;
    const route = parseHash() || { view: 'library' };
    loadChrome();
    pollStatus();
    await go(route, { historyMode: 'none' });
  }

  /* ───────────────────────── 启动 ───────────────────────── */

  async function init() {
    scroller = $('#scroller');
    bodyEl = $('#body');
    await resolveApi();
    applyTheme();
    setZoom(S.zoom);
    updateSegState();
    bind();

    let setup = null;
    try {
      setup = await getJSON('/api/setup');
    } catch (_) { /* 后端未就绪时给出可重试的引导页 */ }
    if (!setup) {
      showWelcome('无法连接本地照片服务，请稍后重试。');
      return;
    }
    S.libraryDir = setup.library || '';
    if (!setup.configured) {
      showWelcome();
      return;
    }
    await enterApp();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
