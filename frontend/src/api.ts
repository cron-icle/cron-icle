const BASE = "/api";

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(BASE + path, {
    method,
    headers: body !== undefined ? { "content-type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let message = res.statusText;
    try {
      const payload = await res.json();
      if (payload?.error) message = payload.error;
    } catch {
      // ignore — fall back to statusText
    }
    throw new Error(message);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

function q(params: Record<string, string | number | null | undefined>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== null && value !== undefined) search.set(key, String(value));
  }
  const s = search.toString();
  return s ? `?${s}` : "";
}

export const listSemanticEvents = <T,>(limit: number, query?: string | null) =>
  req<T>("GET", `/events/semantic${q({ limit, query })}`);
export const listRawEventProcessingOverview = <T,>(limit: number) =>
  req<T>("GET", `/events/raw-overview${q({ limit })}`);
export const getCaptureSettings = <T,>() => req<T>("GET", "/settings/capture");
export const processingQueueStatus = <T,>() => req<T>("GET", "/processing/queue-status");
export const startupDiagnostics = <T,>() => req<T>("GET", "/startup-diagnostics");
export const localAiSetupStatus = <T,>() => req<T>("GET", "/local-ai/setup-status");
export const startCapture = () => req<void>("POST", "/capture/start");
export const stopCapture = () => req<void>("POST", "/capture/stop");
export const setInputPermission = <T,>(input: "mouse" | "keyboard", enabled: boolean) =>
  req<T>("PUT", "/settings/input-permission", { input, enabled });
export const exportData = () => req<string>("GET", "/data/export");
export const deleteAllData = () => req<void>("POST", "/data/delete-all");
export const getDataDirectory = <T,>() => req<T>("GET", "/data-directory");
export const cancelModelDownload = () => req<void>("POST", "/local-ai/cancel-download");
export const setupDownloadChatModel = () => req<void>("POST", "/local-ai/download-chat-model");
export const setupDownloadEmbedModel = () => req<void>("POST", "/local-ai/download-embed-model");
export const setupStartEngine = () => req<void>("POST", "/local-ai/start-engine");
export const setupRemoveChatModel = () => req<void>("DELETE", "/local-ai/chat-model");
export const setupRemoveEmbedModel = () => req<void>("DELETE", "/local-ai/embed-model");
export const changeDataDirectory = () => req<{ restarting: boolean }>("POST", "/data-directory/change");
export const setExcludedApplications = <T,>(applications: string[]) =>
  req<T>("PUT", "/settings/excluded-applications", { applications });
export const setExcludedPaths = <T,>(paths: string[]) => req<T>("PUT", "/settings/excluded-paths", { paths });
export const setKeyboardTextAllowlist = <T,>(applications: string[]) =>
  req<T>("PUT", "/settings/keyboard-text-allowlist", { applications });
export const setWatchedFolders = <T,>(folders: string[]) => req<T>("PUT", "/settings/watched-folders", { folders });
export const setScreenshotPermission = <T,>(enabled: boolean) =>
  req<T>("PUT", "/settings/screenshot-permission", { enabled });
export const retryFailedProcessingTasks = () => req<void>("POST", "/processing/retry-failed");
export const captureDiagnostics = <T,>() => req<T>("GET", "/diagnostics/capture");
export const localAiDownloadProgress = <T,>() => req<T | null>("GET", "/local-ai/progress");
export const dataDirectoryMoveProgress = <T,>() => req<T | null>("GET", "/data-directory/move-progress");

/**
 * Polls a progress endpoint (204 = idle/no payload) at `intervalMs`,
 * replacing the Tauri `listen()` push events this app no longer has once
 * running as a plain HTTP server. Returns a cleanup function.
 */
export function pollProgress<T>(fetcher: () => Promise<T | null>, onUpdate: (value: T | null) => void, intervalMs = 300): () => void {
  let cancelled = false;
  const tick = async () => {
    try {
      const value = await fetcher();
      if (!cancelled) onUpdate(value);
    } catch {
      // transient poll failures are ignored — the next tick retries
    }
  };
  tick();
  const id = setInterval(tick, intervalMs);
  return () => {
    cancelled = true;
    clearInterval(id);
  };
}
