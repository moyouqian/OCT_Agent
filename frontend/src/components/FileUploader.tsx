import { Upload } from "lucide-react";
import { useRef, useState } from "react";
import { apiUrl, uploadMatFile } from "../lib/api";
import type { UploadedFile } from "../types/api";

type Props = {
  onUploaded: (file: UploadedFile) => void;
  onError: (message: string) => void;
};

export function FileUploader({ onUploaded, onError }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);

  async function uploadOne(file: File) {
    if (!file.name.toLowerCase().endsWith(".mat")) {
      onError("当前仅支持上传 .mat 文件。");
      return;
    }
    setIsUploading(true);
    try {
      const uploaded = await uploadMatFile(file);
      onUploaded(uploaded);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      onError(`${detail} 当前后端地址：${apiUrl}`);
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
    <>
      <input
        ref={inputRef}
        type="file"
        accept=".mat"
        hidden
        onChange={(event) => void handleFiles(event.target.files)}
      />
      <button
        className="control-chip"
        type="button"
        title="添加 .mat 附件"
        onClick={() => inputRef.current?.click()}
        disabled={isUploading}
      >
        <Upload size={16} />
        <span>{isUploading ? "上传中…" : "添加文件"}</span>
      </button>
    </>
  );
}
