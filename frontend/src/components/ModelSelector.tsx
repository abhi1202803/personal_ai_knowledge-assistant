import { useEffect, useState } from "react";
import { Settings } from "lucide-react";
import { listModels } from "../api/client";
import type { AvailableModelsResponse } from "../types";

interface Props {
  provider: string;
  model: string;
  onProviderChange: (p: string) => void;
  onModelChange: (m: string) => void;
}

const PROVIDER_LABELS: Record<string, string> = {
  openai: "OpenAI",
  qwen: "Qwen",
  gemini: "Gemini (Google)",
  groq: "Groq",
};

export default function ModelSelector({
  provider,
  model,
  onProviderChange,
  onModelChange,
}: Props) {
  const [models, setModels] = useState<AvailableModelsResponse | null>(null);

  useEffect(() => {
    listModels().then(setModels).catch(console.error);
  }, []);

  const providers = models ? Object.keys(models) : ["groq", "openai", "qwen", "gemini"];
  const chatModels: string[] =
    (models as Record<string, { chat: string[]; embedding: string[] }> | null)?.[provider]
      ?.chat ?? [];

  return (
    <div className="model-selector">
      <div className="section-title">
        <Settings size={16} />
        <span>Model Selection</span>
      </div>

      <label className="field-label">Provider</label>
      <select
        value={provider}
        onChange={(e) => {
          const newProvider = e.target.value;
          onProviderChange(newProvider);
          const newModels =
            (models as Record<string, { chat: string[] }> | null)?.[newProvider]?.chat ?? [];
          if (newModels.length > 0) onModelChange(newModels[0]);
        }}
      >
        {providers.map((p) => (
          <option key={p} value={p}>
            {PROVIDER_LABELS[p] ?? p}
          </option>
        ))}
      </select>

      <label className="field-label">Model</label>
      <select value={model} onChange={(e) => onModelChange(e.target.value)}>
        {chatModels.map((m) => (
          <option key={m} value={m}>
            {m}
          </option>
        ))}
      </select>
    </div>
  );
}
