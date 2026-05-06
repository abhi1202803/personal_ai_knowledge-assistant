import type {
  AvailableModelsResponse,
  ChatRequest,
  DocumentInfo,
  KnowledgeBaseCreate,
  KnowledgeBaseInfo,
  UploadResponse,
} from "../types";

const BASE_URL = "/api";

// -------- Knowledge Base --------

export async function listKnowledgeBases(): Promise<KnowledgeBaseInfo[]> {
  const res = await fetch(`${BASE_URL}/knowledge-bases`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function createKnowledgeBase(
  data: KnowledgeBaseCreate,
): Promise<KnowledgeBaseInfo> {
  const res = await fetch(`${BASE_URL}/knowledge-bases`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || "Failed to create knowledge base");
  }
  return res.json();
}

export async function deleteKnowledgeBase(name: string): Promise<void> {
  const res = await fetch(`${BASE_URL}/knowledge-bases/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function listDocuments(name: string): Promise<DocumentInfo[]> {
  const res = await fetch(
    `${BASE_URL}/knowledge-bases/${encodeURIComponent(name)}/documents`,
  );
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function uploadDocument(
  kbName: string,
  file: File,
  startPage?: number,
  endPage?: number,
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  if (startPage != null) form.append("start_page", String(startPage));
  if (endPage != null) form.append("end_page", String(endPage));

  const res = await fetch(
    `${BASE_URL}/knowledge-bases/${encodeURIComponent(kbName)}/upload`,
    { method: "POST", body: form },
  );
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || "Upload failed");
  }
  return res.json();
}

// -------- Models --------

export async function listModels(): Promise<AvailableModelsResponse> {
  const res = await fetch(`${BASE_URL}/models`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// -------- Chat (SSE streaming) --------

export async function* streamChat(
  req: ChatRequest,
): AsyncGenerator<string, void, unknown> {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    throw new Error(`Chat request failed: ${res.status}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data: ")) continue;

      const jsonStr = trimmed.slice(6);
      let data: { token?: string; done?: boolean; error?: string };
      try {
        data = JSON.parse(jsonStr);
      } catch {
        // skip malformed lines
        continue;
      }

      if (data.error) throw new Error(data.error);
      if (data.done) return;
      if (data.token) yield data.token;
    }
  }
}
