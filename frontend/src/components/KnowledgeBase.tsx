import { useEffect, useState } from "react";
import {
  Database,
  Plus,
  Trash2,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import {
  listKnowledgeBases,
  createKnowledgeBase,
  deleteKnowledgeBase,
  listModels,
} from "../api/client";
import type { AvailableModelsResponse, KnowledgeBaseInfo } from "../types";
import FileUpload from "./FileUpload";

interface Props {
  selectedKb: string | null;
  onSelectKb: (name: string | null) => void;
}

export default function KnowledgeBase({ selectedKb, onSelectKb }: Props) {
  const [kbs, setKbs] = useState<KnowledgeBaseInfo[]>([]);
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [newProvider, setNewProvider] = useState("openai");
  const [newModel, setNewModel] = useState("text-embedding-3-small");
  const [models, setModels] = useState<AvailableModelsResponse | null>(null);
  const [expandedKb, setExpandedKb] = useState<string | null>(null);
  const [error, setError] = useState("");

  const refreshKbs = () => {
    listKnowledgeBases().then(setKbs).catch(console.error);
  };

  useEffect(() => {
    refreshKbs();
    listModels().then(setModels).catch(console.error);
  }, []);

  const allModels = models as Record<string, { chat: string[]; embedding: string[] }> | null;

  const embeddingProviders = allModels
    ? Object.entries(allModels)
        .filter(([, v]) => v.embedding.length > 0)
        .map(([k]) => k)
    : ["openai", "qwen"];

  const embeddingModels = allModels?.[newProvider]?.embedding ?? [];

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setError("");
    try {
      await createKnowledgeBase({
        name: newName.trim(),
        embedding_provider: newProvider,
        embedding_model: newModel,
      });
      setNewName("");
      setShowCreate(false);
      refreshKbs();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleDelete = async (name: string) => {
    if (!confirm(`Delete knowledge base "${name}"? This cannot be undone.`)) return;
    try {
      await deleteKnowledgeBase(name);
      if (selectedKb === name) onSelectKb(null);
      refreshKbs();
    } catch (e: any) {
      setError(e.message);
    }
  };

  return (
    <div className="knowledge-base">
      <div className="section-title">
        <Database size={16} />
        <span>Knowledge Bases</span>
        <button
          className="icon-btn"
          onClick={() => setShowCreate(!showCreate)}
          title="Create knowledge base"
        >
          <Plus size={16} />
        </button>
      </div>

      {showCreate && (
        <div className="create-form">
          <input
            type="text"
            placeholder="Knowledge base name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
          />
          <label className="field-label">Embedding Provider</label>
          <select
            value={newProvider}
            onChange={(e) => {
              setNewProvider(e.target.value);
              const embs = allModels?.[e.target.value]?.embedding ?? [];
              if (embs.length > 0) setNewModel(embs[0]);
            }}
          >
            {embeddingProviders.map((p) => (
              <option key={p} value={p}>
                {p === "openai" ? "OpenAI" : p === "qwen" ? "Qwen" : p}
              </option>
            ))}
          </select>
          <label className="field-label">Embedding Model</label>
          <select
            value={newModel}
            onChange={(e) => setNewModel(e.target.value)}
          >
            {embeddingModels.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <button className="btn btn-primary" onClick={handleCreate}>
            Create
          </button>
          {error && <div className="error-text">{error}</div>}
        </div>
      )}

      <div className="kb-list">
        {kbs.length === 0 && (
          <div className="empty-text">No knowledge bases yet. Click + to create one.</div>
        )}
        {kbs.map((kb) => (
          <div key={kb.name} className="kb-item-wrapper">
            <div
              className={`kb-item ${selectedKb === kb.name ? "active" : ""}`}
              onClick={() =>
                onSelectKb(selectedKb === kb.name ? null : kb.name)
              }
            >
              <span
                className="kb-expand"
                onClick={(e) => {
                  e.stopPropagation();
                  setExpandedKb(expandedKb === kb.name ? null : kb.name);
                }}
              >
                {expandedKb === kb.name ? (
                  <ChevronDown size={14} />
                ) : (
                  <ChevronRight size={14} />
                )}
              </span>
              <span className="kb-name">{kb.name}</span>
              <span className="kb-badge">
                {kb.embedding_provider} - {kb.document_count} chunks
              </span>
              <button
                className="icon-btn danger"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(kb.name);
                }}
                title="Delete"
              >
                <Trash2 size={14} />
              </button>
            </div>

            {expandedKb === kb.name && (
              <div className="kb-expanded">
                <FileUpload kbName={kb.name} onUploaded={refreshKbs} />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
