import { Check, Copy } from "lucide-react";
import { useState, type ReactNode } from "react";

type Props = {
  language: string;
  plainText: string;
  children: ReactNode;
};

export function CodeBlock({ language, plainText, children }: Props) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(plainText);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard 不可用时静默忽略 */
    }
  }

  return (
    <div className="code-block">
      <div className="code-block-header">
        <span>{language || "code"}</span>
        <button
          type="button"
          className={`code-block-copy ${copied ? "copied" : ""}`}
          onClick={copy}
          title="复制代码"
        >
          {copied ? <Check size={13} /> : <Copy size={13} />}
          <span>{copied ? "已复制" : "复制"}</span>
        </button>
      </div>
      <pre>{children}</pre>
    </div>
  );
}
