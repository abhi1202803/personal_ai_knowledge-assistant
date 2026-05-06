import { useEffect, useRef, useState } from "react";
import { Send, Loader2 } from "lucide-react";
import { streamChat } from "../api/client";
import type { ChatMessage } from "../types";
import MessageBubble from "./MessageBubble";
import ImageUpload from "./ImageUpload";

interface Props {
  selectedKb: string | null;
  modelProvider: string;
  modelName: string;
}

export default function ChatPanel({
  selectedKb,
  modelProvider,
  modelName,
}: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [image, setImage] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text && !image) return;
    if (streaming) return;

    const userMsg: ChatMessage = {
      role: "user",
      content: text,
      image: image ?? undefined,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setImage(null);

    const assistantMsg: ChatMessage = { role: "assistant", content: "" };
    setMessages((prev) => [...prev, assistantMsg]);

    setStreaming(true);
    try {
      const gen = streamChat({
        message: text,
        image: image ?? undefined,
        kb_name: selectedKb ?? undefined,
        model_provider: modelProvider,
        model_name: modelName,
      });

      for await (const token of gen) {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === "assistant") {
            updated[updated.length - 1] = {
              ...last,
              content: last.content + token,
            };
          }
          return updated;
        });
      }
    } catch (e: any) {
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last.role === "assistant") {
          updated[updated.length - 1] = {
            ...last,
            content: `Error: ${e.message}`,
          };
        }
        return updated;
      });
    } finally {
      setStreaming(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <h2>AI Assistant</h2>
        <div className="chat-header-info">
          <span className="badge">{modelProvider}/{modelName}</span>
          {selectedKb && (
            <span className="badge badge-kb">Knowledge base: {selectedKb}</span>
          )}
        </div>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>Hello! I am your AI assistant.</p>
            <p>Ask me a question, or create a knowledge base on the left and upload documents first.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}
        {streaming && (
          <div className="streaming-indicator">
            <Loader2 size={14} className="spin" />
            <span>Generating...</span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-area">
        <ImageUpload image={image} onImageChange={setImage} />
        <textarea
          className="chat-input"
          placeholder="Type your question..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
        />
        <button
          className="btn btn-send"
          onClick={handleSend}
          disabled={streaming || (!input.trim() && !image)}
        >
          <Send size={18} />
        </button>
      </div>
    </div>
  );
}
