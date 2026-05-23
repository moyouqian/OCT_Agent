import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { Bot, Play, PlusCircle, Search, Square } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FileUploader } from "./components/FileUploader";
import { MethodPanel } from "./components/MethodPanel";
import { ResultGallery } from "./components/ResultGallery";
import { apiUrl, uploadMatFile } from "./lib/api";
import type { DisplayMessage, MethodSettings, ResultRef, UploadedFile } from "./types/api";

const initialSettings: MethodSettings = {
  visualizationEnabled: true,
  showThinking: false,
  vector: false,
  cnn: false,
  bnn: false,
  Nx: 25,
  Nz: 25,
  g: 1,
  MC_test: 50,
  physical: {
    wavelength: 840e-9,
    bandwidth: 50e-9,
    refractiveIndex: 1.0,
  },
};

function mergeResults(existing: ResultRef[], incoming: ResultRef[]): ResultRef[] {
  const seen = new Set(existing.map((item) => item.result_id));
  const next = [...existing];
  for (const item of incoming) {
    if (item.result_id && !seen.has(item.result_id)) {
      seen.add(item.result_id);
      next.push(item);
    }
  }
  return next;
}

function messageText(message: Message): string {
  if (typeof message.content === "string") return message.content;
  return JSON.stringify(message.content, null, 2);
}

function splitThinking(content: string): { thinking: string | null; answer: string } {
  const match = content.match(/<think>([\s\S]*?)<\/think>/i);
  if (!match) {
    const cleaned = content.replace(/<think>[\s\S]*$/i, "").trim();
    return { thinking: null, answer: cleaned };
  }
  return {
    thinking: match[1].trim(),
    answer: content.replace(match[0], "").trim(),
  };
}

function hasVisibleAiContent(content: string, showThinking: boolean): boolean {
  const { thinking, answer } = splitThinking(content);
  return answer.trim().length > 0 || Boolean(showThinking && thinking);
}

function normalizeMarkdown(content: string): string {
  return content.replace(/\n{3,}/g, "\n\n").trim();
}

function mayGenerateResult(text: string, files: UploadedFile[], settings: MethodSettings): boolean {
  if (files.length > 0) return true;
  if (settings.vector || settings.cnn || settings.bnn) return true;
  return /应变|strain|cnn|bnn|矢量|vector|phase|热力图|\.mat/i.test(text);
}

function ChatView({ onNewConversation }: { onNewConversation: () => void }) {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [settings, setSettings] = useState<MethodSettings>(initialSettings);
  const [input, setInput] = useState("");
  const [results, setResults] = useState<ResultRef[]>([]);
  const [displayMessages, setDisplayMessages] = useState<DisplayMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [deepResearchOnce, setDeepResearchOnce] = useState(false);
  const messageListRef = useRef<HTMLElement | null>(null);
  const messageEndRef = useRef<HTMLDivElement | null>(null);
  const replaceNextResultsRef = useRef(false);
  const savedThreadId = useRef<string | null>(
    localStorage.getItem("oct_thread_id"),
  );

  const fileIds = useMemo(() => files.map((file) => file.file_id), [files]);

  const thread = useStream<{
    messages: Message[];
    file_ids: string[];
    result_refs: ResultRef[];
    strain_settings: unknown;
    physical_params: unknown;
    visualization_enabled: boolean;
    show_thinking: boolean;
    requested_sub_agent?: "deep_research" | null;
    conversation_summary?: string;
  }>({
    apiUrl,
    assistantId: "agent",
    messagesKey: "messages",
    threadId: savedThreadId.current ?? undefined,
    onThreadId: (id: string) => {
      localStorage.setItem("oct_thread_id", id);
      savedThreadId.current = id;
    },
    onUpdateEvent: (event: unknown) => {
      const typed = event as {
        collect_result_refs?: { result_refs?: ResultRef[] };
      };
      const directRefs = typed.collect_result_refs?.result_refs ?? [];
      if (directRefs.length) {
        setResults((current) => {
          if (replaceNextResultsRef.current) {
            replaceNextResultsRef.current = false;
            return directRefs;
          }
          return mergeResults(current, directRefs);
        });
      }
    },
    onError: (err: unknown) => {
      const text = err instanceof Error ? err.message : String(err);
      setError(`运行失败：${text}。当前后端地址：${apiUrl}`);
    },
  });

  useEffect(() => {
    const refs = thread.values?.result_refs ?? [];
    if (refs.length) {
      setResults((current) => {
        if (replaceNextResultsRef.current) {
          replaceNextResultsRef.current = false;
          return refs;
        }
        return mergeResults(current, refs);
      });
    }
  }, [thread.values?.result_refs]);

  useEffect(() => {
    const aiMessages = (thread.messages ?? []).filter((message) => message.type === "ai");
    if (!aiMessages.length) return;
    setDisplayMessages((current) => {
      let next = current;
      for (const message of aiMessages) {
        const content = messageText(message);
        if (!content.trim()) continue;
        const key = message.id ?? content;
        const existingIndex = next.findIndex((item) => item.id === key);
        if (existingIndex >= 0) {
          if (next[existingIndex].content === content) continue;
          next = next.map((item, index) => (index === existingIndex ? { ...item, content } : item));
        } else {
          next = [...next, { id: key, role: "ai", content }];
        }
      }
      return next;
    });
  }, [thread.messages]);

  const handleUploaded = useCallback((file: UploadedFile) => {
    setFiles((current) => [...current, file]);
  }, []);

  const removeFile = useCallback((fileId: string) => {
    setFiles((current) => current.filter((file) => file.file_id !== fileId));
  }, []);

  const uploadDroppedFiles = useCallback(async (droppedFiles: FileList) => {
    const matFile = Array.from(droppedFiles).find((file) => file.name.toLowerCase().endsWith(".mat"));
    if (!matFile) {
      setError("当前仅支持拖入 .mat 文件。");
      return;
    }
    setError(null);
    try {
      const uploaded = await uploadMatFile(matFile);
      setFiles((current) => [...current, uploaded]);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      setError(`${detail} 当前后端地址：${apiUrl}`);
    }
  }, []);

  const submit = useCallback(() => {
    const rawText = input.trim();
    const prompt = rawText || (files.length ? "请处理本轮附件。" : "");
    if (!prompt) return;
    setError(null);
    const requestedSubAgent = deepResearchOnce ? "deep_research" : null;
    const shouldReplaceResults =
      requestedSubAgent === "deep_research" ? false : mayGenerateResult(prompt, files, settings);
    if (shouldReplaceResults) {
      replaceNextResultsRef.current = true;
      setResults([]);
    } else {
      replaceNextResultsRef.current = false;
    }

    const newMessages: Message[] = [
      ...(thread.messages ?? []),
      {
        type: "human",
        content: prompt,
        id: crypto.randomUUID(),
      },
    ];

    setDisplayMessages((current) => [
      ...current,
      {
        id: crypto.randomUUID(),
        role: "human",
        content: rawText || "已发送附件",
      },
    ]);

    thread.submit({
      messages: newMessages,
      file_ids: fileIds,
      result_refs: shouldReplaceResults ? [] : results,
      strain_settings: {
        vector: settings.vector,
        cnn: settings.cnn,
        bnn: settings.bnn,
        Nx: settings.Nx,
        Nz: settings.Nz,
        g: settings.g,
        MC_test: settings.MC_test,
      },
      physical_params: {
        wavelength: settings.physical.wavelength,
        bandwidth: settings.physical.bandwidth,
        refractive_index: settings.physical.refractiveIndex,
      },
      visualization_enabled: settings.visualizationEnabled,
      show_thinking: settings.showThinking,
      requested_sub_agent: requestedSubAgent,
    });

    setInput("");
    setFiles([]);
    setDeepResearchOnce(false);
  }, [deepResearchOnce, fileIds, files, input, results, settings, thread]);

  const canSend = input.trim().length > 0 || files.length > 0;
  const lastMessage = displayMessages[displayMessages.length - 1];
  const showThinkingPlaceholder =
    thread.isLoading &&
    (!lastMessage ||
      lastMessage.role === "human" ||
      (lastMessage.role === "ai" && !hasVisibleAiContent(lastMessage.content, settings.showThinking)));

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ block: "end" });
  }, [displayMessages, showThinkingPlaceholder, thread.isLoading]);

  return (
    <div className="app-shell">
      <main className="chat-shell">
        <header className="chat-header">
          <div className="agent-mark">
            <Bot size={22} />
          </div>
          <div style={{ flex: 1 }}>
            <h1>OCT Agent</h1>
            <p>通用对话、文件分析、应变计算与后续文献总结都会在这里完成。</p>
          </div>
          <button
            type="button"
            className="new-conversation-button"
            onClick={onNewConversation}
            title="开始新对话"
            disabled={thread.isLoading}
          >
            <PlusCircle size={18} />
            <span>新对话</span>
          </button>
        </header>

        <section className="message-list" ref={messageListRef}>
          {displayMessages.length === 0 && !showThinkingPlaceholder ? (
            <div className="welcome">
              <h2>有什么 OCT 相关问题，直接问我。</h2>
              <p>可以先普通聊天，也可以添加 `.mat` 文件后让 Agent 做应变计算。</p>
            </div>
          ) : (
            <>
              {displayMessages.map((message) => (
                <article className={`message ${message.role}`} key={message.id}>
                  <span>{message.role === "human" ? "你" : "OCT Agent"}</span>
                  <MessageContent content={message.content} role={message.role} showThinking={settings.showThinking} />
                </article>
              ))}
              {showThinkingPlaceholder && (
                <article className="message ai thinking-message">
                  <span>OCT Agent</span>
                  <div>思考中...</div>
                </article>
              )}
              <div className="message-end" ref={messageEndRef} />
            </>
          )}
        </section>

        <section
          className={`composer ${isDragging ? "dragging" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setIsDragging(false);
            void uploadDroppedFiles(event.dataTransfer.files);
          }}
        >
          <FileUploader files={files} onUploaded={handleUploaded} onRemove={removeFile} />
          <MethodPanel settings={settings} onChange={setSettings} />
          {error && <p className="error-text">{error}</p>}
          <div className="input-row">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="输入消息，例如：你好；帮我总结 OCT 是什么；对上传的 phase 文件做 CNN 和矢量法应变计算。"
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" &&
                  !event.shiftKey &&
                  !(event.nativeEvent as KeyboardEvent).isComposing
                ) {
                  event.preventDefault();
                  submit();
                }
              }}
            />
            <button
              className={`research-button ${deepResearchOnce ? "active" : ""}`}
              type="button"
              onClick={() => setDeepResearchOnce((current) => !current)}
              disabled={thread.isLoading}
              title="下一条消息使用 Deep Research"
              aria-pressed={deepResearchOnce}
            >
              <Search size={18} />
              <span>Deep Research</span>
            </button>
            <button
              className="run-button"
              type="button"
              onClick={thread.isLoading ? thread.stop : submit}
              disabled={!canSend && !thread.isLoading}
            >
              {thread.isLoading ? <Square size={18} /> : <Play size={18} />}
              <span>{thread.isLoading ? "停止" : "发送"}</span>
            </button>
          </div>
          {isDragging && <div className="drop-hint">松开鼠标即可上传 .mat 文件</div>}
        </section>
      </main>

      <ResultGallery results={results} />
    </div>
  );
}

export default function App() {
  const [sessionKey, setSessionKey] = useState(0);

  const handleNewConversation = useCallback(() => {
    localStorage.removeItem("oct_thread_id");
    setSessionKey((k) => k + 1);
  }, []);

  return <ChatView key={sessionKey} onNewConversation={handleNewConversation} />;
}

function MessageContent({
  content,
  role,
  showThinking,
}: {
  content: string;
  role: "human" | "ai";
  showThinking: boolean;
}) {
  if (role === "human") {
    return <div>{content}</div>;
  }
  const { thinking, answer } = splitThinking(content);
  const rendered = normalizeMarkdown(answer || "思考中...");
  return (
    <div className="markdown-body">
      {thinking && showThinking && (
        <details className="thinking-block" open>
          <summary>深度思考</summary>
          <pre>{thinking}</pre>
        </details>
      )}
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{rendered}</ReactMarkdown>
    </div>
  );
}
