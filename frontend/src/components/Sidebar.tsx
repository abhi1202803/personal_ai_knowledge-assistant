import { Sparkles } from "lucide-react";
import ModelSelector from "./ModelSelector";
import KnowledgeBase from "./KnowledgeBase";

interface Props {
  modelProvider: string;
  modelName: string;
  onProviderChange: (p: string) => void;
  onModelChange: (m: string) => void;
  selectedKb: string | null;
  onSelectKb: (name: string | null) => void;
}

export default function Sidebar({
  modelProvider,
  modelName,
  onProviderChange,
  onModelChange,
  selectedKb,
  onSelectKb,
}: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand-logo" aria-hidden="true">
          <Sparkles size={20} />
        </div>
        <h1>Personal AI Assistant</h1>
      </div>

      <ModelSelector
        provider={modelProvider}
        model={modelName}
        onProviderChange={onProviderChange}
        onModelChange={onModelChange}
      />

      <div className="sidebar-divider" />

      <KnowledgeBase selectedKb={selectedKb} onSelectKb={onSelectKb} />
    </aside>
  );
}
