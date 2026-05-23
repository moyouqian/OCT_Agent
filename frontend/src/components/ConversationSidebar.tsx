import { Client } from "@langchain/langgraph-sdk";
import { MessageSquare, PanelLeftClose, Plus, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { apiUrl } from "../lib/api";

type ThreadSummary = {
  id: string;
  title: string;
  updatedAt: number;
};

const client = new Client({ apiUrl });

function deriveTitle(values: unknown): string {
  const messages = (values as { messages?: Array<{ type?: string; content?: unknown }> })?.messages;
  if (Array.isArray(messages)) {
    const firstHuman = messages.find((m) => m?.type === "human");
    const content = firstHuman?.content;
    const text = typeof content === "string" ? content : Array.isArray(content) ? JSON.stringify(content) : "";
    const trimmed = text.trim();
    if (trimmed) return trimmed.length > 40 ? `${trimmed.slice(0, 40)}…` : trimmed;
  }
  return "新对话";
}

function relativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const min = Math.floor(diff / 60000);
  if (min < 1) return "刚刚";
  if (min < 60) return `${min} 分钟前`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr} 小时前`;
  const day = Math.floor(hr / 24);
  if (day < 30) return `${day} 天前`;
  return new Date(ts).toLocaleDateString("zh-CN");
}

type Props = {
  activeThreadId: string | null;
  refreshKey: number;
  onSelect: (threadId: string) => void;
  onNew: () => void;
  onCollapse: () => void;
  onDeletedActive: () => void;
};

export function ConversationSidebar({
  activeThreadId,
  refreshKey,
  onSelect,
  onNew,
  onCollapse,
  onDeletedActive,
}: Props) {
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

  const load = useCallback(async () => {
    try {
      const result = await client.threads.search({ limit: 50, sortBy: "updated_at", sortOrder: "desc" });
      const summaries: ThreadSummary[] = result
        .map((thread) => ({
          id: thread.thread_id,
          title: deriveTitle(thread.values),
          updatedAt: new Date(thread.updated_at ?? thread.created_at ?? Date.now()).getTime(),
        }))
        .filter((item) => item.title !== "新对话" || item.id === activeThreadId);
      setThreads(summaries);
      setStatus("ready");
    } catch {
      setStatus("error");
    }
  }, [activeThreadId]);

  useEffect(() => {
    void load();
  }, [load, refreshKey]);

  async function handleDelete(event: React.MouseEvent, threadId: string) {
    event.stopPropagation();
    setThreads((current) => current.filter((item) => item.id !== threadId));
    try {
      await client.threads.delete(threadId);
    } catch {
      void load();
    }
    if (threadId === activeThreadId) onDeletedActive();
  }

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <MessageSquare size={18} color="var(--color-primary)" />
        <h1>OCT Agent</h1>
        <button type="button" className="btn-icon" style={{ marginLeft: "auto" }} onClick={onCollapse} title="收起侧栏">
          <PanelLeftClose size={17} />
        </button>
      </div>

      <button type="button" className="btn btn-secondary sidebar-new" onClick={onNew}>
        <Plus size={16} />
        <span>新对话</span>
      </button>

      <div className="sidebar-list">
        <div className="sidebar-section-label">历史会话</div>
        {status === "loading" && <div className="sidebar-loading">正在加载会话…</div>}
        {status === "error" && <div className="sidebar-empty">无法连接后端，暂时无法加载历史会话。</div>}
        {status === "ready" && threads.length === 0 && (
          <div className="sidebar-empty">还没有历史会话，开始一段新对话吧。</div>
        )}
        {threads.map((thread) => (
          <div
            className={`conversation-item ${thread.id === activeThreadId ? "active" : ""}`}
            key={thread.id}
          >
            <button
              type="button"
              className="conversation-open"
              onClick={() => onSelect(thread.id)}
              title={`${thread.title}\n${relativeTime(thread.updatedAt)}`}
            >
              <MessageSquare size={14} style={{ flexShrink: 0, opacity: 0.6 }} />
              <span className="conversation-title">{thread.title}</span>
            </button>
            <button
              type="button"
              className="conversation-delete"
              onClick={(event) => handleDelete(event, thread.id)}
              title="删除会话"
              aria-label="删除会话"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}
      </div>
    </aside>
  );
}
