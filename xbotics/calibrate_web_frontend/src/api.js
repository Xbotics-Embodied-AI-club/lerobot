const API_BASE = '/api';

export class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

async function request(path, { method = 'GET', body } = {}) {
  const options = { method, cache: 'no-store', headers: {} };
  if (body !== undefined) {
    options.headers['Content-Type'] = 'application/json';
    options.body = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch (error) {
    throw new ApiError('无法连接标定服务，请确认后端已启动。', 0, null);
  }

  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { error: text };
    }
  }

  if (!response.ok) {
    const message =
      (data && (data.error || data.detail || data.message)) ||
      `请求失败（HTTP ${response.status}）`;
    throw new ApiError(message, response.status, data);
  }

  return data;
}

export function getDevices() {
  return request('/devices');
}

export function getSession() {
  return request('/session');
}

export function createSession(target, demo, port) {
  return request('/session', { method: 'POST', body: { target, demo: Boolean(demo), port } });
}

export function sendSessionAction(action, joint) {
  const body = { action };
  if (joint) body.joint = joint;
  return request('/session/action', { method: 'POST', body });
}

export function cancelSession() {
  return request('/session/cancel', { method: 'POST', body: {} });
}

export function eventsUrl() {
  return `${API_BASE}/events`;
}
