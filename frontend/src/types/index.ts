// -------- Knowledge Base --------

export interface KnowledgeBaseInfo {
  name: string;
  embedding_provider: string;
  embedding_model: string;
  document_count: number;
}

export interface KnowledgeBaseCreate {
  name: string;
  embedding_provider: string;
  embedding_model: string;
}

export interface DocumentInfo {
  filename: string;
  page_range: string | null;
  chunk_count: number;
}

// -------- Chat --------

export interface ChatRequest {
  message: string;
  image?: string; // base64
  kb_name?: string;
  model_provider: string;
  model_name: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  image?: string; // base64 thumbnail for display
}

// -------- Models --------

export interface ProviderModels {
  chat: string[];
  embedding: string[];
}

export interface AvailableModelsResponse {
  openai: ProviderModels;
  qwen: ProviderModels;
  gemini: ProviderModels;
  groq: ProviderModels;
  [provider: string]: ProviderModels;
}

// -------- Upload --------

export interface UploadResponse {
  message: string;
  filename: string;
  chunk_count: number;
}
