import { useRef } from "react";
import { ImageIcon, X } from "lucide-react";

interface Props {
  image: string | null;
  onImageChange: (base64: string | null) => void;
}

export default function ImageUpload({ image, onImageChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = (file: File | undefined) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const base64 = result.split(",")[1];
      onImageChange(base64);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="image-upload">
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => handleFile(e.target.files?.[0])}
      />

      {image ? (
        <div className="image-preview-wrapper">
          <img
            src={`data:image/png;base64,${image}`}
            alt="preview"
            className="image-preview"
          />
          <button
            className="image-remove"
            onClick={() => {
              onImageChange(null);
              if (inputRef.current) inputRef.current.value = "";
            }}
          >
            <X size={12} />
          </button>
        </div>
      ) : (
        <button
          className="icon-btn"
          onClick={() => inputRef.current?.click()}
          title="Upload image"
        >
          <ImageIcon size={18} />
        </button>
      )}
    </div>
  );
}
