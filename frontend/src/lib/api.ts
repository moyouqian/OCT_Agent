import type { HeatmapPayload, ResultRef, UploadedFile } from "../types/api";

export const apiUrl =
  import.meta.env.VITE_API_URL ??
  (import.meta.env.DEV ? "http://localhost:2024" : window.location.origin);

async function readError(response: Response, url: string): Promise<Error> {
  let detail = "";
  try {
    const data = await response.json();
    detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data);
  } catch {
    detail = await response.text();
  }
  return new Error(`请求失败：${url}，状态码 ${response.status}。${detail || "后端没有返回详细信息。"}`);
}

export async function uploadMatFile(file: File): Promise<UploadedFile> {
  const url = `${apiUrl}/api/files/upload`;
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw await readError(response, url);
  }
  return response.json();
}

export async function fetchResultMetadata(resultId: string): Promise<ResultRef> {
  const url = `${apiUrl}/api/results/${resultId}/metadata`;
  const response = await fetch(url);
  if (!response.ok) {
    throw await readError(response, url);
  }
  return response.json();
}

export async function recoverResultMetadata(result: ResultRef): Promise<ResultRef> {
  const url = `${apiUrl}/api/results/recover`;
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(result),
  });
  if (!response.ok) {
    throw await readError(response, url);
  }
  return response.json();
}

export async function fetchHeatmap(
  resultId: string,
  name = "strain",
): Promise<HeatmapPayload> {
  const url = `${apiUrl}/api/results/${resultId}/array?name=${encodeURIComponent(name)}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw await readError(response, url);
  }
  return response.json();
}

export function downloadUrl(resultId: string): string {
  return `${apiUrl}/api/results/${resultId}/download`;
}
