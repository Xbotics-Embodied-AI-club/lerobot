import './styles.css';
import { CalibrationViewer } from './model-viewer.js';
import {
  JOINT_ORDER,
  STATUS_TEXT,
  PHASE_TEXT,
  ACTION_TEXT,
  jointLabel,
} from './joints.js';
import {
  ApiError,
  getDevices,
  getSession,
  createSession,
  sendSessionAction,
  cancelSession,
  eventsUrl,
} from './api.js';
import { SpeechPlayer } from './speech.js';

const $ = (id) => document.getElementById(id);

const state = {
  devices: null,
  serialPorts: [],
  demoDefault: false,
  snapshot: null,
  sessionId: null,
  lastSeq: -1,
  lastPhase: null,
  dismissedSessionId: null,
  actionPending: false,
  connectPending: false,
  modelStatus: 'loading',
  stream: null,
  streamFailures: 0,
  pollTimer: null,
  toastTimer: null,
  lastActiveJoint: null,
  reviewVisible: false,
  actionsKey: '',
  renderTimer: null,
};

let viewer = null;
const speechPlayer = new SpeechPlayer({ onError: (message) => showToast(message, 'error') });

const TARGET_TEXT = { leader: '主臂', follower: '从臂' };

const PHASE_INSTRUCTION = {
  middle: '将机械臂摆到中位姿态，确认六个关节都处于自然零点。',
  ranges: '保持设备在线，缓慢转动当前关节到两端极限并确认。',
  review: '检查每个关节的候选范围与回零偏移，确认无误后写入标定。',
  writing: '正在将标定参数写入电机，请勿断开设备电源。',
  done: '标定参数已写入，机械臂可以开始使用。',
  error: '标定流程出现异常，请查看错误信息后重试或取消会话。',
  cancelled: '标定会话已取消，可以重新连接开始新的标定。',
};

const PHASE_TITLE = {
  middle: '调整到中位姿态',
  ranges: '记录关节限位',
  review: '复核标定结果',
  writing: '正在写入标定',
  done: '标定完成',
  error: '标定出现异常',
  cancelled: '会话已取消',
};

const STEP_INDEX = { middle: 1, ranges: 2, review: 3, writing: 4, done: 5 };

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => {
    const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
    return map[char];
  });
}

function isSnapshot(value) {
  return Boolean(value && typeof value === 'object' && value.id && value.phase);
}

function num(value) {
  return value === null || value === undefined || value === '' ? null : Number(value);
}

function fmtNum(value, digits = 0) {
  const parsed = num(value);
  if (parsed === null || !Number.isFinite(parsed)) return '—';
  return parsed.toFixed(digits);
}

function fmtRange(min, max, digits = 0) {
  const low = num(min);
  const high = num(max);
  if (low === null && high === null) return '—';
  return `${low === null ? '—' : low.toFixed(digits)}–${high === null ? '—' : high.toFixed(digits)}`;
}

function fmtAngle(jointName, value) {
  const parsed = num(value);
  if (parsed === null || !Number.isFinite(parsed)) return '—';
  return jointName === 'gripper' ? `${parsed.toFixed(1)}%` : `${parsed.toFixed(1)}°`;
}

function targetLabel(target) {
  return TARGET_TEXT[target] || target || '未知设备';
}

function phaseText(phase) {
  return PHASE_TEXT[phase] || '等待会话';
}

function effectivePhase(snapshot) {
  if (!snapshot) return null;
  if (snapshot.phase === 'error' || snapshot.phase === 'cancelled') {
    return state.lastPhase || null;
  }
  return snapshot.phase;
}

function showToast(message, kind = 'info') {
  const toast = $('toast');
  toast.textContent = message;
  toast.className = `toast ${kind}`;
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => {
    toast.classList.add('hidden');
  }, 3800);
}

function setViewerMessage(kind, text) {
  const box = $('viewerMessage');
  box.classList.toggle('hidden', kind === 'ready');
  box.dataset.state = kind;
  box.replaceChildren();

  if (kind === 'loading') {
    const spinner = document.createElement('div');
    spinner.className = 'spinner';
    const label = document.createElement('p');
    label.textContent = text || '正在加载模型…';
    box.append(spinner, label);
    return;
  }

  if (kind === 'error') {
    const icon = document.createElement('div');
    icon.className = 'viewer-message-icon';
    icon.textContent = '!';
    const title = document.createElement('h3');
    title.textContent = '3D 模型加载失败';
    const retry = document.createElement('button');
    retry.type = 'button';
    retry.className = 'btn small ghost';
    retry.textContent = '重试';
    retry.addEventListener('click', () => viewer?.load());
    box.append(icon, title, retry);
  }
}

function handleModelStatus({ status, text }) {
  state.modelStatus = status;
  setViewerMessage(status, text);
  updateViewerButtons();
}

function updateViewerButtons() {
  const ready = state.modelStatus === 'ready';
  $('togglePlay').disabled = !ready;
  $('resetView').disabled = !ready;
  $('focusJoint').disabled = !ready || !state.snapshot?.active_joint;
  $('togglePlay').textContent = viewer?.playing ? '暂停目标动画' : '继续目标动画';
}

function hideOverlay() {
  $('connectOverlay').classList.add('hidden');
}

function showOverlay() {
  $('connectOverlay').classList.remove('hidden');
}

function setConnectError(message) {
  const box = $('connectError');
  box.textContent = message || '';
  box.classList.toggle('hidden', !message);
}

function setDevicesError(message) {
  const box = $('devicesError');
  box.textContent = message || '';
  box.classList.toggle('hidden', !message);
}

function renderDeviceOptions(devices) {
  for (const target of ['follower', 'leader']) {
    const element = $(`${target}Info`);
    const device = devices?.[target];
    if (!device) {
      element.textContent = '未检测到设备';
      continue;
    }
    const parts = [device.port, device.id ? `ID ${device.id}` : null, device.type].filter(Boolean);
    element.textContent = parts.join(' · ') || '设备信息未知';
  }
  updatePortOptions();
}

function updatePortOptions() {
  const target = document.querySelector('input[name="target"]:checked')?.value || 'follower';
  const defaultPort = state.devices?.[target]?.port;
  const devices = [...new Set([defaultPort, ...state.serialPorts.map((port) => port.device)].filter(Boolean))];
  const select = $('portSelect');
  select.innerHTML = devices.length
    ? devices.map((device) => {
        const detected = state.serialPorts.find((port) => port.device === device);
        const suffix = device === defaultPort ? '默认' : detected?.description || '已检测串口';
        return `<option value="${esc(device)}">${esc(device)} · ${esc(suffix)}</option>`;
      }).join('')
    : '<option value="">未发现可用串口</option>';
  if (defaultPort) select.value = defaultPort;
  const demo = $('demoToggle').checked;
  select.disabled = demo || devices.length === 0;
}

async function loadDevices() {
  const loading = $('devicesLoading');
  try {
    const data = await getDevices();
    state.devices = data?.devices || data || null;
    state.serialPorts = Array.isArray(data?.ports) ? data.ports : [];
    if (typeof data?.demo_default === 'boolean') {
      state.demoDefault = data.demo_default;
      if (!state.snapshot) $('demoToggle').checked = state.demoDefault;
    }
    loading.classList.add('hidden');
    setDevicesError('');
    renderDeviceOptions(state.devices);
    renderDeviceSummary();
  } catch (error) {
    loading.classList.add('hidden');
    state.serialPorts = [];
    renderDeviceOptions(null);
    setDevicesError(error instanceof ApiError ? error.message : '检测设备端口失败。');
  }
}

function renderDeviceSummary() {
  const snapshot = state.snapshot;
  const target = snapshot?.target || document.querySelector('input[name="target"]:checked')?.value;
  const device = target ? state.devices?.[target] : null;
  $('summaryTarget').textContent = snapshot ? targetLabel(target) : '未连接';
  $('summaryPort').textContent = snapshot?.port || device?.port || '—';
  const demo = snapshot?.demo ?? $('demoToggle').checked;
  $('summaryDemo').textContent = snapshot ? (demo ? '演示模式' : '实机模式') : demo ? '演示模式' : '实机模式';
  $('summaryDemo').classList.toggle('demo', Boolean(demo));
}

function renderSteps() {
  const snapshot = state.snapshot;
  const phase = snapshot?.phase;
  const current = phase === 'error' || phase === 'cancelled' ? effectivePhase(snapshot) : phase;
  const currentIndex = STEP_INDEX[current] ?? -1;
  const terminalError = phase === 'error' || phase === 'cancelled';

  document.querySelectorAll('#stepList li').forEach((item) => {
    const step = item.dataset.step;
    const index = step === 'connect' ? 0 : STEP_INDEX[step];
    item.classList.remove('active', 'done', 'error', 'cancelled');
    if (!snapshot) {
      if (step === 'connect') item.classList.add('active');
      return;
    }
    if (step === 'connect') {
      item.classList.toggle('done', Boolean(snapshot.connected));
      if (!snapshot.connected) item.classList.add('active');
      return;
    }
    if (index < currentIndex) {
      item.classList.add('done');
    } else if (index === currentIndex) {
      item.classList.add(terminalError ? (phase === 'error' ? 'error' : 'cancelled') : 'active');
      if (phase === 'done') {
        item.classList.remove('active');
        item.classList.add('done');
      }
    } else if (phase === 'done') {
      item.classList.add('done');
    }
  });

  const phaseBadge = $('phaseBadge');
  phaseBadge.textContent = snapshot ? phaseText(phase) : '待连接';
  phaseBadge.className = `phase-badge phase-${snapshot ? phase : 'idle'}`;

  const errorBox = $('footerError');
  const message = snapshot?.error || '';
  errorBox.textContent = message;
  errorBox.classList.toggle('hidden', !message);

  renderReview(snapshot);
}

function renderReview(snapshot) {
  const section = $('reviewSection');
  const phase = snapshot?.phase;
  const shouldShow = Boolean(snapshot) && ['review', 'writing', 'done'].includes(phase);
  if (!shouldShow) {
    section.classList.add('hidden');
    state.reviewVisible = false;
    return;
  }
  const joints = snapshot.joints || {};
  const tbody = $('reviewTable').querySelector('tbody');
  tbody.innerHTML = JOINT_ORDER.map((name) => {
    const joint = joints[name] || {};
    const status = joint.status || 'pending';
    return `<tr>
      <td>
        <span class="review-joint">
          <i class="dot ${status}"></i>
          <span><strong>${esc(jointLabel(name, joint.label))}</strong></span>
        </span>
      </td>
      <td>${fmtRange(joint.range_min, joint.range_max)}</td>
      <td>${fmtNum(joint.homing_offset, 0)}</td>
    </tr>`;
  }).join('');
  section.classList.remove('hidden');
  if (phase === 'review' && !state.reviewVisible) {
    const panel = section.closest('.steps-panel');
    if (panel) {
      const top = section.getBoundingClientRect().top - panel.getBoundingClientRect().top + panel.scrollTop - 12;
      panel.scrollTop = top;
    }
  }
  state.reviewVisible = true;
}

function renderInstruction() {
  const snapshot = state.snapshot;
  const phase = snapshot?.phase;
  const title = snapshot ? PHASE_TITLE[phase] || phaseText(phase) : '等待连接';
  const active = snapshot?.active_joint;
  const activeLabel = active ? jointLabel(active, snapshot.joints?.[active]?.label) : '';

  let heading = title;
  if (phase === 'ranges' && active) heading = `${activeLabel} · 完整行程`;

  $('instructionTitle').textContent = heading;
  $('instructionText').textContent = snapshot
    ? snapshot.instruction || PHASE_INSTRUCTION[phase] || PHASE_INSTRUCTION.middle
    : '';

  // 可执行的操作已经是底栏的按钮；这里只在出错时说明原因
  const hint = $('actionHint');
  hint.textContent = snapshot?.error || '';
  hint.className = snapshot?.error ? 'muted error-text' : 'muted';

  const card = $('instructionCard');
  card.className = `instruction-card phase-${phase || 'idle'}`;
}

function renderJoints() {
  const list = $('jointList');
  const snapshot = state.snapshot;
  const joints = snapshot?.joints || {};
  const active = snapshot?.active_joint;
  const canRetry = (snapshot?.available_actions || []).includes('retry_joint');

  list.innerHTML = JOINT_ORDER.map((name, index) => {
    const joint = joints[name] || {};
    const status = joint.status || 'pending';
    const label = jointLabel(name, joint.label);
    const isActive = name === active;
    const reason = joint.reason
      ? `<p class="joint-reason">${esc(joint.reason)}</p>`
      : '';
    const retry =
      canRetry && snapshot.phase === 'review' && name !== 'wrist_roll'
        ? `<button type="button" class="joint-retry" data-joint="${esc(name)}" ${
            state.actionPending ? 'disabled' : ''
          }>重试</button>`
        : '';
    return `<article class="joint-card ${status} ${isActive ? 'is-active' : ''}" data-joint="${esc(name)}">
      <header class="joint-card-head">
        <span class="joint-index">${index + 1}</span>
        <span class="joint-name"><strong>${esc(label)}</strong></span>
        <span class="joint-status status-${status}"><i class="dot ${status}"></i>${esc(
          STATUS_TEXT[status] || status,
        )}</span>
        ${retry}
      </header>
      <div class="joint-metrics">
        <div><span>原始值</span><strong>${fmtNum(joint.raw)}</strong></div>
        <div><span>角度</span><strong>${fmtAngle(name, joint.angle)}</strong></div>
        <div><span>观测范围</span><strong>${fmtRange(joint.observed_min, joint.observed_max)}</strong></div>
        <div><span>候选范围</span><strong>${fmtRange(joint.range_min, joint.range_max)}</strong></div>
        <div><span>回零偏移</span><strong>${fmtNum(joint.homing_offset)}</strong></div>
      </div>
      ${reason}
    </article>`;
  }).join('');

  if (active && active !== state.lastActiveJoint) {
    const card = list.querySelector(`[data-joint="${CSS.escape(active)}"]`);
    card?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  state.lastActiveJoint = active || null;

  const passCount = JOINT_ORDER.filter((name) => joints[name]?.status === 'pass').length;
  $('jointPassCount').textContent = snapshot ? `${passCount} / ${JOINT_ORDER.length}` : '—';
}

function actionLabel(action, snapshot) {
  if (action === 'confirm_limit') return '记录限位';
  if (action === 'retry_joint') {
    const active = snapshot?.active_joint;
    if (active) return `重试 ${jointLabel(active, snapshot.joints?.[active]?.label)}`;
    return '重试关节';
  }
  return ACTION_TEXT[action] || action;
}

function actionClass(action) {
  if (action === 'write') return 'btn primary';
  if (action === 'retry_joint') return 'btn warn';
  return 'btn accent';
}

function renderActions() {
  const snapshot = state.snapshot;
  const actions = [...new Set(snapshot?.available_actions || [])];
  const order = ['confirm_middle', 'confirm_limit', 'retry_joint', 'write'];
  actions.sort((a, b) => {
    const ai = order.indexOf(a);
    const bi = order.indexOf(b);
    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
  });

  const key = [
    snapshot?.id || 'none',
    snapshot?.phase || 'none',
    snapshot?.stage || '',
    snapshot?.active_joint || '',
    actions.join(','),
    state.actionPending ? 'pending' : 'idle',
  ].join('|');
  if (key === state.actionsKey) return;
  state.actionsKey = key;

  const container = $('actionButtons');
  container.replaceChildren();

  for (const action of actions) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = actionClass(action);
    button.textContent = actionLabel(action, snapshot);
    button.disabled = state.actionPending;
    button.dataset.action = action;
    button.addEventListener('click', () => runAction(action));
    container.append(button);
  }

  if (snapshot && !actions.length) {
    const span = document.createElement('span');
    span.className = 'muted action-placeholder';
    span.textContent =
      snapshot.phase === 'done'
        ? '标定流程已完成'
        : snapshot.phase === 'cancelled'
          ? '会话已取消'
          : '暂无可执行操作';
    container.append(span);
  }

  const cancel = $('cancelButton');
  const canCancel = snapshot && !['done', 'cancelled'].includes(snapshot.phase);
  cancel.classList.toggle('hidden', !canCancel);
  cancel.disabled = state.actionPending;

  const newSession = $('newSessionButton');
  const canStartNew = snapshot && ['done', 'cancelled'].includes(snapshot.phase);
  newSession.classList.toggle('hidden', !canStartNew);
  newSession.disabled = state.actionPending;
}

function renderAll() {
  renderDeviceSummary();
  renderSteps();
  renderInstruction();
  renderJoints();
  renderActions();
  updateViewerButtons();
}

function resetSessionState({ dismissed = true } = {}) {
  if (dismissed && state.sessionId) state.dismissedSessionId = state.sessionId;
  state.snapshot = null;
  state.sessionId = null;
  state.lastSeq = -1;
  state.lastPhase = null;
  state.lastActiveJoint = null;
  $('demoToggle').checked = state.demoDefault;
  updatePortOptions();
  stopEventStream();
  viewer?.clear();
  showOverlay();
  renderAll();
}

function applySnapshot(snapshot, source = 'sse') {
  if (!isSnapshot(snapshot)) return;
  if (state.dismissedSessionId && snapshot.id === state.dismissedSessionId) return;

  if (state.sessionId === snapshot.id) {
    const seq = num(snapshot.seq);
    if (seq !== null && state.lastSeq >= 0 && seq < state.lastSeq) return;
    if (seq !== null) state.lastSeq = seq;
  } else {
    state.sessionId = snapshot.id;
    state.lastSeq = num(snapshot.seq) ?? 0;
    state.lastPhase = null;
  }

  state.snapshot = snapshot;
  if (snapshot.phase !== 'error' && snapshot.phase !== 'cancelled') {
    state.lastPhase = snapshot.phase;
  }

  hideOverlay();
  speechPlayer.speak(snapshot.speech_instruction || snapshot.instruction);
  viewer?.setSnapshot(snapshot);
  if (source === 'sse' || source === 'poll') {
    scheduleRender();
  } else {
    if (state.renderTimer) {
      clearTimeout(state.renderTimer);
      state.renderTimer = null;
    }
    renderAll();
  }
}

function scheduleRender() {
  if (state.renderTimer) return;
  state.renderTimer = setTimeout(() => {
    state.renderTimer = null;
    renderAll();
  }, 150);
}

async function refreshSession(source = 'poll') {
  try {
    const snapshot = await getSession();
    if (isSnapshot(snapshot)) {
      applySnapshot(snapshot, source);
      return snapshot;
    }
    return null;
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      handleNoSession();
      return null;
    }
    throw error;
  }
}

function handleNoSession() {
  if (state.snapshot) {
    resetSessionState();
    showToast('标定会话已结束，请重新连接。', 'info');
  }
}

async function attachExistingSession() {
  try {
    const snapshot = await getSession();
    if (isSnapshot(snapshot)) {
      applySnapshot(snapshot, 'initial');
      startEventStream();
    }
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 404) {
      console.warn('获取现有会话失败', error);
    }
  }
}

async function connect() {
  if (state.connectPending) return;
  speechPlayer.unlock();
  const target = document.querySelector('input[name="target"]:checked')?.value || 'follower';
  const demo = $('demoToggle').checked;
  const port = $('portSelect').value;
  const button = $('connectButton');
  if (!demo && !port) {
    setConnectError('请选择要连接的串口。');
    return;
  }

  state.connectPending = true;
  button.disabled = true;
  button.textContent = '正在连接…';
  setConnectError('');
  setDevicesError('');

  try {
    const result = await createSession(target, demo, port);
    state.dismissedSessionId = null;
    if (isSnapshot(result)) {
      applySnapshot(result, 'connect');
    } else {
      const snapshot = await getSession();
      applySnapshot(snapshot, 'connect');
    }
    hideOverlay();
    startEventStream({ force: true });
    showToast(`已连接${targetLabel(target)}，开始标定`, 'success');
  } catch (error) {
    if (error instanceof ApiError && error.status === 409) {
      try {
        const snapshot = await getSession();
        if (isSnapshot(snapshot)) {
          state.dismissedSessionId = null;
          applySnapshot(snapshot, 'connect');
          hideOverlay();
          startEventStream({ force: true });
          showToast('已有标定会话，已接入当前会话。', 'info');
          return;
        }
      } catch (innerError) {
        console.warn('接入现有会话失败', innerError);
      }
      setConnectError('已有标定会话正在运行，请先取消现有会话。');
    } else {
      setConnectError(error?.message || '连接失败，请稍后重试。');
    }
  } finally {
    state.connectPending = false;
    button.disabled = false;
    button.textContent = '开始标定';
  }
}

async function runAction(action, jointOverride = null) {
  if (!state.snapshot || state.actionPending) return;

  let joint;
  if (action === 'retry_joint') {
    const joints = state.snapshot.joints || {};
    joint =
      jointOverride ||
      state.snapshot.active_joint ||
      Object.keys(joints).find(
        (name) => name !== 'wrist_roll' && joints[name]?.status === 'error',
      );
    if (!joint) {
      showToast('请在关节卡片上选择要重试的关节。', 'info');
      return;
    }
  }

  state.actionPending = true;
  renderActions();
  renderJoints();

  try {
    const result = await sendSessionAction(action, joint);
    if (isSnapshot(result)) {
      applySnapshot(result, 'action');
    } else {
      await refreshSession('action');
    }
    $('footerError').classList.add('hidden');
  } catch (error) {
    const message = error?.message || '操作失败';
    showToast(message, 'error');
    const footerError = $('footerError');
    footerError.textContent = message;
    footerError.classList.remove('hidden');
  } finally {
    state.actionPending = false;
    renderActions();
    renderJoints();
  }
}

async function startNewSession() {
  if (!state.snapshot || state.actionPending) return;
  state.actionPending = true;
  renderActions();
  const sessionId = state.sessionId;
  try {
    if (state.snapshot.phase !== 'cancelled') {
      try {
        await cancelSession();
      } catch (error) {
        if (!(error instanceof ApiError) || error.status !== 404) throw error;
      }
    }
  } catch (error) {
    showToast(error?.message || '结束当前会话失败', 'error');
    state.actionPending = false;
    renderActions();
    return;
  }
  state.dismissedSessionId = sessionId;
  state.actionPending = false;
  resetSessionState({ dismissed: false });
  setConnectError('');
  renderActions();
}

async function cancelCurrent() {
  if (!state.snapshot || state.actionPending) return;
  state.actionPending = true;
  renderActions();
  const sessionId = state.sessionId;

  try {
    await cancelSession();
    state.dismissedSessionId = sessionId;
    resetSessionState({ dismissed: false });
    setConnectError('');
    showToast('已取消标定会话，可以重新连接。', 'info');
  } catch (error) {
    showToast(error?.message || '取消会话失败', 'error');
  } finally {
    state.actionPending = false;
    renderActions();
  }
}

function setStreamStatus(mode, text) {
  // 实时通道正常时不占位置，断了、重连中才露出来
  const indicator = $('streamIndicator');
  indicator.className = `status-pill ${mode}`;
  indicator.classList.toggle('hidden', mode === 'online' || mode === 'neutral');
  indicator.querySelector('span').textContent = text;
}

function startEventStream({ force = false } = {}) {
  if (state.stream) {
    if (!force && state.stream.readyState !== EventSource.CLOSED) return;
    state.stream.close();
    state.stream = null;
  }
  if (!('EventSource' in window)) {
    setStreamStatus('offline', '浏览器不支持实时通道');
    startPolling();
    return;
  }

  setStreamStatus('connecting', '实时通道连接中…');
  try {
    const source = new EventSource(eventsUrl());
    state.stream = source;

    source.onopen = () => {
      state.streamFailures = 0;
      setStreamStatus('online', '实时通道已连接');
      stopPolling();
    };

    source.onmessage = (event) => {
      state.streamFailures = 0;
      setStreamStatus('online', '实时通道已连接');
      stopPolling();
      let snapshot;
      try {
        snapshot = JSON.parse(event.data);
      } catch (error) {
        console.warn('解析 SSE 消息失败', error);
        return;
      }
      applySnapshot(snapshot, 'sse');
    };

    source.onerror = () => {
      state.streamFailures += 1;
      if (source.readyState === EventSource.CLOSED) {
        state.stream = null;
        setStreamStatus('offline', '实时通道已断开，使用轮询');
        startPolling();
      } else {
        setStreamStatus('reconnecting', '实时通道重连中…');
        if (state.streamFailures >= 3) startPolling();
      }
    };
  } catch (error) {
    console.warn('创建实时通道失败', error);
    state.stream = null;
    setStreamStatus('offline', '实时通道不可用，使用轮询');
    startPolling();
  }
}

function stopEventStream() {
  if (state.stream) {
    state.stream.close();
    state.stream = null;
  }
  stopPolling();
  state.streamFailures = 0;
  setStreamStatus('neutral', '等待会话');
}

function startPolling() {
  if (state.pollTimer) return;
  state.pollTimer = setInterval(async () => {
    try {
      await refreshSession('poll');
    } catch (error) {
      console.warn('轮询会话失败', error);
    }
  }, 2000);
}

function stopPolling() {
  if (!state.pollTimer) return;
  clearInterval(state.pollTimer);
  state.pollTimer = null;
}

function renderSpeechToggle() {
  const button = $('speechToggle');
  button.textContent = `语音：${speechPlayer.enabled ? '开' : '关'}`;
  button.setAttribute('aria-pressed', String(speechPlayer.enabled));
}

function handleConfirmShortcut(event) {
  if (event.key !== 'Enter' || event.repeat || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
  if (!state.snapshot || state.actionPending || state.connectPending) return;
  if (event.target instanceof Element && event.target.closest('button, input, textarea, select, [contenteditable="true"]')) return;

  const actionByPhase = { middle: 'confirm_middle', ranges: 'confirm_limit' };
  const action = actionByPhase[state.snapshot.phase];
  if (!action || !state.snapshot.available_actions.includes(action)) return;
  event.preventDefault();
  runAction(action);
}

function bindEvents() {
  renderSpeechToggle();
  document.addEventListener('keydown', handleConfirmShortcut);
  document.addEventListener('pointerdown', () => speechPlayer.unlock(), { once: true });
  $('speechToggle').addEventListener('click', () => {
    speechPlayer.setEnabled(!speechPlayer.enabled);
    speechPlayer.unlock();
    renderSpeechToggle();
  });
  $('connectButton').addEventListener('click', connect);
  $('cancelButton').addEventListener('click', cancelCurrent);
  $('newSessionButton').addEventListener('click', startNewSession);
  $('demoToggle').addEventListener('change', () => {
    updatePortOptions();
    if (!state.snapshot) renderDeviceSummary();
  });
  document.querySelectorAll('input[name="target"]').forEach((input) => {
    input.addEventListener('change', () => {
      updatePortOptions();
      if (!state.snapshot) renderDeviceSummary();
    });
  });
  $('togglePlay').addEventListener('click', () => {
    viewer?.togglePlaying();
    updateViewerButtons();
  });
  $('jointList').addEventListener('click', (event) => {
    const button = event.target.closest('.joint-retry');
    if (!button) return;
    runAction('retry_joint', button.dataset.joint);
  });
  $('resetView').addEventListener('click', () => viewer?.resetView());
  $('focusJoint').addEventListener('click', () => {
    const active = state.snapshot?.active_joint;
    if (!active) {
      showToast('当前没有需要聚焦的关节。', 'info');
      return;
    }
    if (!viewer?.focusJoint(active)) {
      showToast('3D 模型尚未就绪，无法聚焦关节。', 'error');
    }
  });
}

function init() {
  bindEvents();
  setViewerMessage('loading', '正在加载模型…');
  viewer = new CalibrationViewer($('viewer'), { onStatus: handleModelStatus });
  viewer.load();

  renderAll();
  loadDevices();
  stopEventStream();
  attachExistingSession();
}

init();
