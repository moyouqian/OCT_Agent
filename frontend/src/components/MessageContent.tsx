import { Bot, Check, Copy, User } from "lucide-react";
import {
  isValidElement,
  useState,
  type ComponentPropsWithoutRef,
  type ReactElement,
  type ReactNode,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { CodeBlock } from "./CodeBlock";

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

export function hasVisibleAiContent(content: string, showThinking: boolean): boolean {
  const { thinking, answer } = splitThinking(content);
  return answer.trim().length > 0 || Boolean(showThinking && thinking);
}

function normalizeMarkdown(content: string): string {
  return content.replace(/\n{3,}/g, "\n\n").trim();
}

function extractText(node: ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (isValidElement(node)) {
    return extractText((node.props as { children?: ReactNode }).children);
  }
  return "";
}

const markdownComponents = {
  pre({ children }: ComponentPropsWithoutRef<"pre">) {
    const codeEl = children as ReactElement<{ className?: string; children?: ReactNode }>;
    const className = (isValidElement(codeEl) && codeEl.props.className) || "";
    const match = /language-(\w+)/.exec(className);
    const language = match?.[1] ?? "";
    const plainText = extractText(children);
    return (
      <CodeBlock language={language} plainText={plainText}>
        {children}
      </CodeBlock>
    );
  },
};

export function MessageContent({
  content,
  role,
  showThinking,
}: {
  content: string;
  role: "human" | "ai";
  showThinking: boolean;
}) {
  const [copied, setCopied] = useState(false);

  async function copyMessage(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard 不可用时静默忽略 */
    }
  }

  if (role === "human") {
    return (
      <>
        <div className="message-meta">
          <span className="message-avatar">
            <User size={15} />
          </span>
          <span>你</span>
        </div>
        <div className="message-bubble">{content}</div>
      </>
    );
  }

  const { thinking, answer } = splitThinking(content);
  const rendered = normalizeMarkdown(answer || "思考中...");
  const copyPayload = thinking ? `${answer}`.trim() : rendered;

  return (
    <>
      <div className="message-meta">
        <span className="message-avatar">
          <Bot size={15} />
        </span>
        <span>OCT Agent</span>
      </div>
      <div className="message-bubble">
        <button
          type="button"
          className={`message-copy ${copied ? "copied" : ""}`}
          onClick={() => copyMessage(copyPayload || rendered)}
          title="复制回复"
        >
          {copied ? <Check size={13} /> : <Copy size={13} />}
          <span>{copied ? "已复制" : "复制"}</span>
        </button>
        <div className="markdown-body">
          {thinking && showThinking && (
            <details className="thinking-block" open>
              <summary>深度思考</summary>
              <pre>{thinking}</pre>
            </details>
          )}
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
            components={markdownComponents}
          >
            {rendered}
          </ReactMarkdown>
        </div>
      </div>
    </>
  );
}
