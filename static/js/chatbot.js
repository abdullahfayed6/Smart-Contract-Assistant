const API_BASE = "";
const REQUEST_TIMEOUT_MS = 180000;

const state = {
  sessionId: newSessionId(),
  docs: [],
  selectedDocId: "",
};

const el = {
  healthBadge: document.getElementById("health-badge"),
  sessionBadge: document.getElementById("session-badge"),
  statusAlert: document.getElementById("status-alert"),
  uploadForm: document.getElementById("upload-form"),
  fileInput: document.getElementById("file-input"),
  uploadBtn: document.getElementById("upload-btn"),
  docSelect: document.getElementById("doc-select"),
  docInfo: document.getElementById("doc-info"),
  refreshBtn: document.getElementById("refresh-btn"),
  newChatBtn: document.getElementById("new-chat-btn"),
  clearBtn: document.getElementById("clear-btn"),
  chatLog: document.getElementById("chat-log"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  sendBtn: document.getElementById("send-btn"),
};

function newSessionId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  const seed = Math.random().toString(16).slice(2);
  return `${Date.now()}-${seed}`;
}

function shortSessionId(id) {
  return String(id || "").slice(0, 8);
}

function setSessionBadge() {
  el.sessionBadge.textContent = `Session: ${shortSessionId(state.sessionId) || "-"}`;
}

function setStatus(message, tone = "secondary") {
  if (!message) {
    el.statusAlert.className = "alert alert-secondary mt-3 mb-0 d-none";
    el.statusAlert.textContent = "";
    return;
  }
  el.statusAlert.className = `alert alert-${tone} mt-3 mb-0`;
  el.statusAlert.textContent = message;
}

function setButtonBusy(button, busy, busyText) {
  if (!button) {
    return;
  }
  if (!button.dataset.defaultText) {
    button.dataset.defaultText = button.textContent.trim();
  }
  button.disabled = busy;
  button.textContent = busy ? busyText : button.dataset.defaultText;
}

function renderDocOptions(preferredDocId = "") {
  const docs = state.docs || [];
  const choices = docs.map((row) => String(row.doc_id || "")).filter(Boolean);

  if (choices.length === 0) {
    state.selectedDocId = "";
    el.docSelect.innerHTML = '<option value="">No documents indexed</option>';
    el.docSelect.value = "";
    el.docSelect.disabled = true;
    renderDocInfo();
    return;
  }

  const nextSelected =
    choices.includes(preferredDocId) ? preferredDocId :
    choices.includes(state.selectedDocId) ? state.selectedDocId :
    choices[0];

  state.selectedDocId = nextSelected;
  el.docSelect.disabled = false;
  el.docSelect.innerHTML = "";
  for (const docId of choices) {
    const option = document.createElement("option");
    option.value = docId;
    option.textContent = docId;
    el.docSelect.appendChild(option);
  }
  el.docSelect.value = nextSelected;
  renderDocInfo();
}

function renderDocInfo() {
  if (!state.selectedDocId) {
    el.docInfo.textContent = "No active document selected.";
    return;
  }
  const doc = (state.docs || []).find((row) => String(row.doc_id || "") === state.selectedDocId);
  if (!doc) {
    el.docInfo.textContent = "Selected document is no longer available.";
    return;
  }
  const filename = String(doc.filename || "");
  const sections = Number(doc.pages_or_sections || 0);
  const chunks = Number(doc.chunks_indexed || 0);
  el.docInfo.textContent = `File: ${filename} | sections: ${sections} | chunks: ${chunks}`;
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function assistantMarkdownToHtml(text) {
  const raw = compactChunkIdReferences(String(text || ""));

  if (window.marked && window.DOMPurify) {
    marked.setOptions({ gfm: true, breaks: true });
    const rendered = marked.parse(raw);
    return DOMPurify.sanitize(rendered);
  }

  // Fallback when CDN libs fail: keep safe text with visible line breaks.
  return escapeHtml(raw).replaceAll("\n", "<br>");
}

function compactChunkIdReferences(text) {
  return text.replace(/chunk_id\s*=\s*([A-Za-z0-9._:-]+)/gi, (fullMatch, rawId) => {
    const id = String(rawId || "");
    const fromChunkSuffix = id.match(/(?:^|_c)(\d+)$/i);
    if (fromChunkSuffix) {
      return fullMatch.replace(rawId, fromChunkSuffix[1]);
    }
    const numericOnly = id.match(/^c?(\d+)$/i);
    if (numericOnly) {
      return fullMatch.replace(rawId, numericOnly[1]);
    }
    return fullMatch;
  });
}

function addMessage(role, content, metaText = "") {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant") {
    bubble.classList.add("markdown-bubble");
    bubble.innerHTML = assistantMarkdownToHtml(content || "");
  } else {
    bubble.textContent = content || "";
  }

  if (metaText) {
    const meta = document.createElement("div");
    meta.className = "bubble-meta";
    meta.textContent = metaText;
    bubble.appendChild(meta);
  }

  row.appendChild(bubble);
  el.chatLog.appendChild(row);
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
}

function resetChat(welcomeMessage) {
  el.chatLog.innerHTML = "";
  addMessage("assistant", welcomeMessage);
}

function apiUrl(path) {
  return `${API_BASE}${path}`;
}

async function apiFetch(path, init = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(apiUrl(path), { ...init, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function parseApiError(response) {
  try {
    const payload = await response.json();
    if (payload && typeof payload === "object" && payload.detail) {
      return String(payload.detail);
    }
  } catch (_) {
    // Ignore parsing failures and fallback to plain text.
  }
  try {
    const text = await response.text();
    return text || `HTTP ${response.status}`;
  } catch (_) {
    return `HTTP ${response.status}`;
  }
}

async function refreshDocuments(preferredDocId = "", successMessage = "") {
  try {
    const response = await apiFetch("/api/documents", { method: "GET" }, 20000);
    if (!response.ok) {
      setStatus(`Failed to load documents: ${await parseApiError(response)}`, "danger");
      return;
    }
    const payload = await response.json();
    state.docs = Array.isArray(payload) ? payload : [];
    renderDocOptions(preferredDocId);
    if (successMessage) {
      setStatus(successMessage, "success");
    } else {
      setStatus(`Documents refreshed. Found ${state.docs.length} document(s).`, "secondary");
    }
  } catch (error) {
    setStatus(`Failed to load documents: ${String(error)}`, "danger");
  }
}

async function loadHealth() {
  try {
    const response = await apiFetch("/api/health", { method: "GET" }, 15000);
    if (!response.ok) {
      throw new Error(await parseApiError(response));
    }
    const payload = await response.json();
    const model = payload?.chat_model ? String(payload.chat_model) : "unknown";
    el.healthBadge.className = "badge text-bg-success";
    el.healthBadge.innerHTML = '<i class="bi bi-dot"></i>API Online';
    el.healthBadge.title = `Chat model: ${model}`;
  } catch (_) {
    el.healthBadge.className = "badge text-bg-danger";
    el.healthBadge.innerHTML = '<i class="bi bi-dot"></i>API Offline';
    el.healthBadge.title = "Could not reach /api/health";
  }
}

async function handleUpload(event) {
  event.preventDefault();
  const file = el.fileInput.files && el.fileInput.files[0];
  if (!file) {
    setStatus("Select a PDF or DOCX file first.", "warning");
    return;
  }

  const lowerName = file.name.toLowerCase();
  if (!(lowerName.endsWith(".pdf") || lowerName.endsWith(".docx"))) {
    setStatus("Only PDF and DOCX files are supported.", "warning");
    return;
  }

  setButtonBusy(el.uploadBtn, true, "Uploading...");
  try {
    const body = new FormData();
    body.append("file", file);
    const response = await apiFetch(
      `/api/documents/upload?session_id=${encodeURIComponent(state.sessionId)}`,
      { method: "POST", body },
    );

    if (!response.ok) {
      setStatus(`Upload failed: ${await parseApiError(response)}`, "danger");
      return;
    }

    const payload = await response.json();
    const uploadedDocId = String(payload.doc_id || "");
    const chunks = Number(payload.chunks_indexed || 0);
    state.sessionId = newSessionId();
    setSessionBadge();
    resetChat(`Indexed "${file.name}" successfully (${chunks} chunks). Ask your first question.`);
    await refreshDocuments(uploadedDocId, `Uploaded and indexed "${file.name}".`);
    el.fileInput.value = "";
  } catch (error) {
    setStatus(`Upload failed: ${String(error)}`, "danger");
  } finally {
    setButtonBusy(el.uploadBtn, false, "Uploading...");
  }
}

async function handleSend(event) {
  event.preventDefault();
  const text = (el.chatInput.value || "").trim();
  if (!text) {
    return;
  }

  addMessage("user", text);
  el.chatInput.value = "";

  if (!state.selectedDocId) {
    addMessage("assistant", "Upload and select a document first.");
    return;
  }

  setButtonBusy(el.sendBtn, true, "Sending...");
  try {
    const response = await apiFetch(
      `/api/chat/${encodeURIComponent(state.sessionId)}/${encodeURIComponent(state.selectedDocId)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      },
    );

    if (!response.ok) {
      addMessage("assistant", `Error: ${await parseApiError(response)}`);
      return;
    }

    const payload = await response.json();
    const answer = String(payload.answer || "").trim() || "No answer returned.";
    const confidence = Number(payload.confidence || 0).toFixed(2);
    const latencyMs = Number(payload.latency_ms || 0);
    const retrieved = Number(payload.retrieved_chunks || 0);
    const meta = `confidence=${confidence} | latency=${latencyMs} ms | retrieved=${retrieved}`;
    addMessage("assistant", answer, meta);
  } catch (error) {
    addMessage("assistant", `Error: request failed: ${String(error)}`);
  } finally {
    setButtonBusy(el.sendBtn, false, "Sending...");
    el.chatInput.focus();
  }
}

function handleNewChat() {
  state.sessionId = newSessionId();
  setSessionBadge();
  if (state.selectedDocId) {
    resetChat("New chat started. Ask a question about the selected document.");
  } else {
    resetChat("Upload a contract to start chatting.");
  }
  setStatus("Started a new chat session.", "secondary");
}

async function handleClearData() {
  const confirmed = window.confirm("Clear all indexed documents and stored uploads?");
  if (!confirmed) {
    return;
  }

  setButtonBusy(el.clearBtn, true, "Clearing...");
  try {
    const response = await apiFetch("/api/documents/clear", { method: "POST" }, 60000);
    if (!response.ok) {
      setStatus(`Clear failed: ${await parseApiError(response)}`, "danger");
      return;
    }
    state.sessionId = newSessionId();
    setSessionBadge();
    resetChat("Storage cleared. Upload a new document to continue.");
    await refreshDocuments("", "All indexed documents and stored uploads were cleared.");
  } catch (error) {
    setStatus(`Clear failed: ${String(error)}`, "danger");
  } finally {
    setButtonBusy(el.clearBtn, false, "Clearing...");
  }
}

function wireEvents() {
  el.uploadForm.addEventListener("submit", handleUpload);
  el.refreshBtn.addEventListener("click", () => refreshDocuments());
  el.newChatBtn.addEventListener("click", handleNewChat);
  el.clearBtn.addEventListener("click", handleClearData);
  el.docSelect.addEventListener("change", () => {
    state.selectedDocId = el.docSelect.value || "";
    renderDocInfo();
  });
  el.chatForm.addEventListener("submit", handleSend);
  el.chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      el.chatForm.requestSubmit();
    }
  });
}

async function init() {
  setSessionBadge();
  resetChat("Upload a contract to start chatting.");
  wireEvents();
  await Promise.all([loadHealth(), refreshDocuments()]);
}

init();
