import { useState } from "react";
import Sidebar from "./components/Sidebar";
import ChatPanel from "./components/ChatPanel";

const LS_PROVIDER = "pa_model_provider";
const LS_MODEL = "pa_model_name";

export default function App() {
  const [modelProvider, setModelProvider] = useState(
    () => localStorage.getItem(LS_PROVIDER) ?? "groq"
  );
  const [modelName, setModelName] = useState(
    () => localStorage.getItem(LS_MODEL) ?? "llama-3.3-70b-versatile"
  );
  const [selectedKb, setSelectedKb] = useState<string | null>(null);

  const handleProviderChange = (p: string) => {
    setModelProvider(p);
    localStorage.setItem(LS_PROVIDER, p);
  };

  const handleModelChange = (m: string) => {
    setModelName(m);
    localStorage.setItem(LS_MODEL, m);
  };

  return (
    <div className="app">
      <Sidebar
        modelProvider={modelProvider}
        modelName={modelName}
        onProviderChange={handleProviderChange}
        onModelChange={handleModelChange}
        selectedKb={selectedKb}
        onSelectKb={setSelectedKb}
      />
      <ChatPanel
        selectedKb={selectedKb}
        modelProvider={modelProvider}
        modelName={modelName}
      />
    </div>
  );
}
