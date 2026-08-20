import * as api from "./api";

type Diagnostics = { settings: { enabled: boolean; mouse_enabled: boolean; keyboard_enabled: boolean; screenshots_enabled: boolean; excluded_applications: string[]; watched_folders: string[] }; storage: Record<string, number>; queue: Record<string, number>; providers: { semantic_provider: string; embedding_provider: string; semantic_available: boolean; embedding_available: boolean } };

function mountDiagnosticsPanel() {
  const button = document.createElement("button");
  button.className = "quiet-button diagnostics-toggle";
  button.textContent = "Diagnostics";
  document.body.append(button);
  const panel = document.createElement("aside");
  panel.className = "diagnostics-panel";
  panel.hidden = true;
  document.body.append(panel);
  button.onclick = async () => {
    panel.hidden = !panel.hidden;
    if (panel.hidden) return;
    panel.innerHTML = "<h2>Capture diagnostics</h2><p>Loading…</p>";
    try {
      const diagnostics = await api.captureDiagnostics<Diagnostics>();
      panel.innerHTML = `<div class="section-heading"><h2>Capture diagnostics</h2><button class="quiet-button">Close</button></div><p><strong>Capture:</strong> ${diagnostics.settings.enabled ? "enabled" : "disabled"} · <strong>Mouse:</strong> ${diagnostics.settings.mouse_enabled ? "on" : "off"} · <strong>Keyboard:</strong> ${diagnostics.settings.keyboard_enabled ? "on" : "off"} · <strong>Screen:</strong> ${diagnostics.settings.screenshots_enabled ? "on" : "off"}</p><p><strong>Watched folders:</strong> ${diagnostics.settings.watched_folders.length} · <strong>Excluded applications:</strong> ${diagnostics.settings.excluded_applications.length}</p><p><strong>Storage:</strong> ${Object.entries(diagnostics.storage).map(([name, count]) => `${name} ${count}`).join(" · ")}</p><p><strong>Queue:</strong> ${Object.entries(diagnostics.queue).map(([name, count]) => `${name} ${count}`).join(" · ")}</p><p><strong>Semantic:</strong> ${diagnostics.providers.semantic_provider} (${diagnostics.providers.semantic_available ? "available" : "unavailable"}) · <strong>Embeddings:</strong> ${diagnostics.providers.embedding_provider} (${diagnostics.providers.embedding_available ? "available" : "unavailable"})</p>`;
      panel.querySelector("button")?.addEventListener("click", () => { panel.hidden = true; });
    } catch (error) { panel.innerHTML = `<h2>Capture diagnostics</h2><p>Diagnostics unavailable: ${String(error)}</p>`; }
  };
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mountDiagnosticsPanel); else mountDiagnosticsPanel();
