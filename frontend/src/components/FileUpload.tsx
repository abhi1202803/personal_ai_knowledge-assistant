import { useRef, useState } from "react";
import { Upload, FileText } from "lucide-react";
import { uploadDocument } from "../api/client";

interface Props {
  kbName: string;
  onUploaded: () => void;
}

export default function FileUpload({ kbName, onUploaded }: Props) {
  const fileRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [startPage, setStartPage] = useState("");
  const [endPage, setEndPage] = useState("");
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState("");

  const isPdf = file?.name.toLowerCase().endsWith(".pdf");

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setMessage("");
    try {
      const sp = startPage ? parseInt(startPage, 10) : undefined;
      const ep = endPage ? parseInt(endPage, 10) : undefined;
      const res = await uploadDocument(kbName, file, sp, ep);
      setMessage(`Success: ${res.message}`);
      setFile(null);
      setStartPage("");
      setEndPage("");
      if (fileRef.current) fileRef.current.value = "";
      onUploaded();
    } catch (e: any) {
      setMessage(`Error: ${e.message}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="file-upload">
      <div className="upload-area">
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.html,.htm"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          id={`file-${kbName}`}
          hidden
        />
        <label htmlFor={`file-${kbName}`} className="upload-label">
          <Upload size={16} />
          <span>{file ? file.name : "Choose a PDF or HTML file"}</span>
        </label>
      </div>

      {isPdf && (
        <div className="page-range">
          <FileText size={14} />
          <span>Page range:</span>
          <input
            type="number"
            min={1}
            placeholder="Start page"
            value={startPage}
            onChange={(e) => setStartPage(e.target.value)}
          />
          <span>-</span>
          <input
            type="number"
            min={1}
            placeholder="End page"
            value={endPage}
            onChange={(e) => setEndPage(e.target.value)}
          />
        </div>
      )}

      <button
        className="btn btn-primary"
        onClick={handleUpload}
        disabled={!file || uploading}
      >
        {uploading ? "Uploading..." : "Upload to knowledge base"}
      </button>

      {message && (
        <div
          className={`upload-message ${message.startsWith("Error") ? "error" : "success"}`}
        >
          {message}
        </div>
      )}
    </div>
  );
}
