import { Paperclip, Upload, X } from "lucide-react";
import { useRef, useState } from "react";
import { apiUrl, uploadMatFile } from "../lib/api";
import type { UploadedFile } from "../types/api";

type Props = {
  files: UploadedFile[];
  onUploaded: (file: UploadedFile) => void;
  onRemove: (fileId: string) => void;
};

export function FileUploader({ files, onUploaded, onRemove }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function uploadOne(file: File) {
    if (!file.name.toLowerCase().endsWith(".mat")) {
      setError("当前仅支持上传 .mat 文件。");
      return;
    }
    setIsUploading(true);
    setError(null);
    try {
      const uploaded = await uploadMatFile(file);
      onUploaded(uploaded);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      setError(`${detail} 当前后端地址：${apiUrl}`);
    } finally {
      setIsUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  async function handleFiles(selected: FileList | null) {
    const file = selected?.[0];
    if (!file) return;
    await uploadOne(file);
  }

  return (
    <div className="attachment-area">
      <input
        ref={inputRef}
        type="file"
        accept=".mat"
        hidden
        onChange={(event) => void handleFiles(event.target.files)}
      />
      <div className="attachment-row">
        <button
          className="attachment-button"
          type="button"
          title="添加附件"
          onClick={() => inputRef.current?.click()}
          disabled={isUploading}
        >
          <Upload size={17} />
          <span>{isUploading ? "上传中" : "添加文件"}</span>
        </button>
        {files.map((file) => (
          <span className="attachment-chip" key={file.file_id} title={file.original_name}>
            <Paperclip size={14} />
            <span>{file.original_name}</span>
            <button type="button" title="移除附件" onClick={() => onRemove(file.file_id)}>
              <X size={13} />
            </button>
          </span>
        ))}
      </div>
      {error && <p className="error-text">{error}</p>}
    </div>
  );
}
