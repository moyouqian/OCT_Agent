import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import {
  Bot,
  ChevronDown,
  Database,
  FlaskConical,
  PanelLeft,
  Paperclip,
  Play,
  Search,
  SlidersHorizontal,
  Square,
  X,
} from "lucide-react";
import type { ComponentType } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ConversationSidebar } from "./components/ConversationSidebar";
import { FileUploader } from "./components/FileUploader";
import { MessageContent, hasVisibleAiContent } from "./components/MessageContent";
import { MethodPanel } from "./components/MethodPanel";
import { ResultGallery } from "./components/ResultGallery";
import { WelcomeScreen } from "./components/WelcomeScreen";
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

function mayGenerateResult(text: string, files: UploadedFile[], settings: MethodSettings): boolean {
  if (files.length > 0) return true;
  if (settings.vector || settings.cnn || settings.bnn) return true;
  return /应变|strain|cnn|bnn|矢量|vector|phase|热力图|\.mat/i.test(text);
}

// 与后端 requested_sub_agent 字面量对应，用于显式路由到对应子图。
type RequestedAgent = "strain_estimation" | "deep_research" | "self_rag";

const SUBAGENTS: { id: RequestedAgent; label: string; icon: ComponentType<{ size?: number }> }[] = [
  { id: "strain_estimation", label: "应变计算", icon: FlaskConical },
  { id: "deep_research", label: "Deep Research", icon: Search },
  { id: "self_rag", label: "知识库", icon: Database },
];

type ChatViewProps = {
  threadId: string | null;
  onThreadCreated: (id: string) => void;
  onExpandSidebar: () => void;
  sidebarCollapsed: boolean;
  panelHidden: boolean;
  onTogglePanel: () => void;
};

function ChatView({
  threadId,
  onThreadCreated,
  onExpandSidebar,
  sidebarCollapsed,
  panelHidden,
  onTogglePanel,
}: ChatViewProps) {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [settings, setSettings] = useState<MethodSettings>(initialSettings);
  const [input, setInput] = useState("");
  const [results, setResults] = useState<ResultRef[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [requestedAgent, setRequestedAgent] = useState<RequestedAgent | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const messageEndRef = useRef<HTMLDivElement | null>(null);
  const replaceNextResultsRef = useRef(false);

  const fileIds = useMemo(() => files.map((file) => file.file_id), [files]);

  const thread = useStream<{
    messages: Message[];
    file_ids: string[];
    result_refs: ResultRef[];
    strain_settings: unknown;
    physical_params: unknown;
    visualization_enabled: boolean;
    show_thinking: boolean;
    requested_sub_agent?: RequestedAgent | null;
    conversation_summary?: string;
  }>({
    apiUrl,
    assistantId: "agent",
    messagesKey: "messages",
    threadId: threadId ?? undefined,
    onThreadId: (id: string) => {
      onThreadCreated(id);
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

  // 计算结果可能从 thread 状态回填（如刷新或切换会话后）。
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

  // 对话消息直接从 thread.messages 派生，保证切换历史会话时能完整还原。
  const displayMessages = useMemo<DisplayMessage[]>(() => {
    const out: DisplayMessage[] = [];
    for (const message of thread.messages ?? []) {
      if (message.type !== "human" && message.type !== "ai") continue;
      const content = messageText(message);
      if (!content.trim()) continue;
      out.push({ id: message.id ?? `msg-${out.length}`, role: message.type, content });
    }
    return out;
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
    const requestedSubAgent = requestedAgent;
    let shouldReplaceResults: boolean;
    if (requestedSubAgent === "strain_estimation") {
      shouldReplaceResults = true;
    } else if (requestedSubAgent) {
      // deep_research / self_rag 不产生热力图结果，保留既有结果。
      shouldReplaceResults = false;
    } else {
      shouldReplaceResults = mayGenerateResult(prompt, files, settings);
    }
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
    setRequestedAgent(null);
  }, [requestedAgent, fileIds, files, input, results, settings, thread]);

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

  const isEmptyConversation = displayMessages.length === 0 && !showThinkingPlaceholder;

  return (
    <>
      <main className="chat-shell">
        <header className="chat-header">
          {sidebarCollapsed && (
            <button type="button" className="btn-icon" onClick={onExpandSidebar} title="展开侧栏">
              <PanelLeft size={18} />
            </button>
          )}
          <div className="agent-mark">
            <Bot size={20} />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <h1>OCT Agent</h1>
            <p>通用对话、文件分析、应变计算与文献研究都在这里完成。</p>
          </div>
        </header>

        <section className="message-list">
          {isEmptyConversation ? (
            <WelcomeScreen onPick={setInput} />
          ) : (
            <>
              {displayMessages.map((message) => (
                <article className={`message ${message.role}`} key={message.id}>
                  <MessageContent content={message.content} role={message.role} showThinking={settings.showThinking} />
                </article>
              ))}
              {showThinkingPlaceholder && (
                <article className="message ai thinking-message">
                  <div className="message-meta">
                    <span className="message-avatar">
                      <Bot size={15} />
                    </span>
                    <span>OCT Agent</span>
                  </div>
                  <div className="message-bubble">
                    <span className="typing-dots">
                      <span />
                      <span />
                      <span />
                    </span>
                  </div>
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
          <div className="composer-controls">
            <FileUploader onUploaded={handleUploaded} onError={setError} />
            <button
              type="button"
              className={`control-chip ${advancedOpen ? "active" : ""}`}
              onClick={() => setAdvancedOpen((current) => !current)}
              aria-expanded={advancedOpen}
              title="应变计算高级设置"
            >
              <SlidersHorizontal size={16} />
              <span>高级设置</span>
              <ChevronDown size={14} className={`chevron ${advancedOpen ? "open" : ""}`} />
            </button>
            <div className="control-spacer" />
            <div className="subagent-toggles">
              {SUBAGENTS.map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  type="button"
                  className={`subagent-btn ${requestedAgent === id ? "active" : ""}`}
                  onClick={() => setRequestedAgent((current) => (current === id ? null : id))}
                  disabled={thread.isLoading}
                  aria-pressed={requestedAgent === id}
                  title={`下一条消息显式路由到「${label}」子图`}
                >
                  <Icon size={16} />
                  <span>{label}</span>
                </button>
              ))}
            </div>
          </div>

          {files.length > 0 && (
            <div className="attachment-chips">
              {files.map((file) => (
                <span className="attachment-chip" key={file.file_id} title={file.original_name}>
                  <Paperclip size={14} />
                  <span>{file.original_name}</span>
                  <button type="button" title="移除附件" onClick={() => removeFile(file.file_id)}>
                    <X size={13} />
                  </button>
                </span>
              ))}
            </div>
          )}

          {advancedOpen && (
            <div className="advanced-panel">
              <MethodPanel settings={settings} onChange={setSettings} />
            </div>
          )}

          {error && <p className="error-text">{error}</p>}

          <div className="input-row">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="输入消息，或拖入 .mat 文件让 Agent 做应变计算。Enter 发送，Shift+Enter 换行。"
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
              className="btn btn-primary run-button"
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

      {!panelHidden && <ResultGallery results={results} onHide={onTogglePanel} />}
      {panelHidden && (
        <button type="button" className="panel-reopen" onClick={onTogglePanel} title="展开结果看板">
          <Search size={15} />
          <span>结果看板</span>
        </button>
      )}
    </>
  );
}

export default function App() {
  const [activeThreadId, setActiveThreadId] = useState<string | null>(
    localStorage.getItem("oct_thread_id"),
  );
  const [sessionKey, setSessionKey] = useState(0);
  const [sidebarRefresh, setSidebarRefresh] = useState(0);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [panelHidden, setPanelHidden] = useState(false);

  const handleNewConversation = useCallback(() => {
    localStorage.removeItem("oct_thread_id");
    setActiveThreadId(null);
    setSessionKey((k) => k + 1);
    setSidebarRefresh((k) => k + 1);
  }, []);

  const handleSelectThread = useCallback(
    (id: string) => {
      if (id === activeThreadId) return;
      localStorage.setItem("oct_thread_id", id);
      setActiveThreadId(id);
      setSessionKey((k) => k + 1);
    },
    [activeThreadId],
  );

  const handleThreadCreated = useCallback((id: string) => {
    localStorage.setItem("oct_thread_id", id);
    setActiveThreadId(id);
    setSidebarRefresh((k) => k + 1);
  }, []);

  const shellClass = [
    "app-shell",
    sidebarCollapsed ? "sidebar-collapsed" : "",
    panelHidden ? "panel-hidden" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={shellClass}>
      {!sidebarCollapsed && (
        <ConversationSidebar
          activeThreadId={activeThreadId}
          refreshKey={sidebarRefresh}
          onSelect={handleSelectThread}
          onNew={handleNewConversation}
          onCollapse={() => setSidebarCollapsed(true)}
          onDeletedActive={handleNewConversation}
        />
      )}
      <ChatView
        key={sessionKey}
        threadId={activeThreadId}
        onThreadCreated={handleThreadCreated}
        onExpandSidebar={() => setSidebarCollapsed(false)}
        sidebarCollapsed={sidebarCollapsed}
        panelHidden={panelHidden}
        onTogglePanel={() => setPanelHidden((value) => !value)}
      />
    </div>
  );
}
