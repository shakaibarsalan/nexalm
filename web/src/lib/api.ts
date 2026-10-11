import type {
  AdminUser,
  AuthStatus,
  AuthUser,
  ChatSettings,
  Conversation,
  ConversationMetrics,
  Message,
  ModelInfo,
  SessionStatus,
  WorkerPool,
  WorkerSample,
} from './types';

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    credentials: 'same-origin',
    headers: init?.body ? { 'Content-Type': 'application/json' } : undefined,
    ...init,
  });

  if (!res.ok) {
    let code = String(res.status);
    let message = res.statusText;
    try {
      const body = await res.json();
      code = body.error ?? code;
      message = body.message ?? message;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, code, message);
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

export const api = {
  authStatus: () => request<AuthStatus>('/api/auth/status'),

  // 202 with verificationRequired means no account was made yet: a code went
  // to the address and has to come back before the account exists.
  signup: (email: string, password: string, displayName: string) =>
    request<{
      user?: AuthUser;
      adoptedConversations?: number;
      verificationRequired?: boolean;
      email?: string;
      expiresInMinutes?: number;
      message?: string;
    }>('/api/auth/signup', {
      method: 'POST',
      body: JSON.stringify({ email, password, displayName }),
    }),

  verifyEmail: (email: string, code: string) =>
    request<{ user: AuthUser; adoptedConversations: number }>('/api/auth/verify-email', {
      method: 'POST',
      body: JSON.stringify({ email, code }),
    }),

  resendCode: (email: string) =>
    request<{ ok: boolean; expiresInMinutes: number; sentThisHour: number }>(
      '/api/auth/resend-code',
      { method: 'POST', body: JSON.stringify({ email }) },
    ),

  login: (email: string, password: string) =>
    request<{ user: AuthUser; adoptedConversations: number }>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    }),

  logout: () => request<{ ok: true }>('/api/auth/logout', { method: 'POST', body: JSON.stringify({}) }),

  listUsers: () => request<{ users: AdminUser[] }>('/api/users').then((r) => r.users),

  updateUser: (id: string, patch: { role?: 'admin' | 'user'; disabled?: boolean }) =>
    request<{ user: AdminUser }>(`/api/users/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }),

  deleteUser: (id: string) => request<{ ok: true }>(`/api/users/${id}`, { method: 'DELETE' }),

  unlock: (token: string) =>
    request<{ ok: true }>('/api/auth/unlock', {
      method: 'POST',
      body: JSON.stringify({ token }),
    }),

  lock: () => request<{ ok: true }>('/api/auth/lock', { method: 'POST' }),

  listConversations: () =>
    request<{ conversations: Conversation[] }>('/api/conversations').then((r) => r.conversations),

  createConversation: () =>
    request<{ conversation: Conversation }>('/api/conversations', {
      method: 'POST',
      body: JSON.stringify({}),
    }).then((r) => r.conversation),

  getConversation: (id: string) =>
    request<{ conversation: Conversation; messages: Message[] }>(`/api/conversations/${id}`),

  updateConversation: (id: string, patch: { title?: string; settings?: ChatSettings; archived?: boolean }) =>
    request<{ conversation: Conversation }>(`/api/conversations/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    }).then((r) => r.conversation),

  deleteConversation: (id: string) =>
    request<{ ok: true }>(`/api/conversations/${id}`, { method: 'DELETE' }),

  conversationMetrics: (id: string) =>
    request<{ metrics: ConversationMetrics }>(`/api/conversations/${id}/metrics`).then((r) => r.metrics),

  sendMessage: (conversationId: string, content: string) =>
    request<{
      userMessage: Message;
      assistantMessage: Message;
      /** Null when the turn was queued because no GPU could serve it yet. */
      jobId: string | null;
      queued?: boolean;
      message?: string;
      /** Set when the pool had no worker for the chosen model and one is loading. */
      loadingModel?: string;
      /** The model that will actually answer. */
      model?: string;
      /** Set when no model was chosen and the pool answered with what it had. */
      substitutedFor?: string;
    }>(
      `/api/conversations/${conversationId}/messages`,
      { method: 'POST', body: JSON.stringify({ content }) },
    ),

  retry: (conversationId: string) =>
    request<{ assistantMessageId: string; jobId: string }>(`/api/conversations/${conversationId}/retry`, {
      method: 'POST',
      body: JSON.stringify({}),
    }),

  cancelJob: (jobId: string) => request<{ ok: true }>(`/api/jobs/${jobId}/cancel`, { method: 'POST' }),

  getSettings: () =>
    request<{ defaults: ChatSettings; models: ModelInfo[] }>('/api/settings'),

  saveSettings: (defaults: ChatSettings) =>
    request<{ defaults: ChatSettings }>('/api/settings', {
      method: 'PUT',
      body: JSON.stringify(defaults),
    }).then((r) => r.defaults),

  workers: () => request<WorkerPool>('/api/workers'),

  workerSamples: (minutes = 15) =>
    request<{ windowMinutes: number; samples: WorkerSample[] }>(
      `/api/workers/samples?minutes=${minutes}`,
    ),

  sessionStatus: () => request<SessionStatus>('/api/workers/session'),

  models: () =>
    request<{ models: ModelInfo[]; largestVramGb: number | null }>('/api/models'),

  startWorker: (platform: 'kaggle' | 'colab' = 'kaggle') =>
    request<{ ok: boolean; message: string; manual?: boolean; url?: string }>(
      '/api/workers/start',
      { method: 'POST', body: JSON.stringify({ platform }) },
    ),

  // A worker id is required once more than one can be connected; without it the
  // server refuses rather than stopping the whole pool.
  stopWorker: (workerId?: string) =>
    request<{ ok: boolean; message: string; stopped: string[] }>('/api/workers/stop', {
      method: 'POST',
      body: JSON.stringify(workerId ? { workerId } : { all: true }),
    }),

  loadModel: (modelId: string, workerId?: string) =>
    request<{ ok: boolean; message: string; workerId?: string }>('/api/workers/models/load', {
      method: 'POST',
      body: JSON.stringify({ modelId, ...(workerId ? { workerId } : {}) }),
    }),

  unloadModel: (modelId: string, workerId?: string) =>
    request<{ ok: boolean; message: string }>('/api/workers/models/unload', {
      method: 'POST',
      body: JSON.stringify({ modelId, ...(workerId ? { workerId } : {}) }),
    }),
};
