import { Download, Maximize2, Minimize2 } from "lucide-react";
import { useEffect, useState } from "react";
import { downloadUrl, fetchResultMetadata, recoverResultMetadata } from "../lib/api";
import type { ResultRef } from "../types/api";
import { HeatmapViewer } from "./HeatmapViewer";

type Props = {
  results: ResultRef[];
};

export function ResultGallery({ results }: Props) {
  const [expanded, setExpanded] = useState(false);

  if (results.length === 0) {
    return (
      <aside className={`artifact-panel empty ${expanded ? "expanded" : ""}`}>
        <PanelTitle expanded={expanded} onToggle={() => setExpanded((value) => !value)} />
        <p>计算结果、热力图、文献卡片和报告草稿会出现在这里。</p>
      </aside>
    );
  }
  return (
    <aside className={`artifact-panel ${expanded ? "expanded" : ""}`}>
      <PanelTitle expanded={expanded} onToggle={() => setExpanded((value) => !value)} />
      <div className="artifact-list">
        {results.map((result) => (
          <ResultCard expanded={expanded} key={result.result_id} result={result} />
        ))}
      </div>
    </aside>
  );
}

function PanelTitle({ expanded, onToggle }: { expanded: boolean; onToggle: () => void }) {
  return (
    <header className="artifact-panel-header">
      <h2>结果</h2>
      <button className="icon-button" onClick={onToggle} title={expanded ? "收起结果面板" : "展开结果面板"} type="button">
        {expanded ? <Minimize2 size={17} /> : <Maximize2 size={17} />}
      </button>
    </header>
  );
}

function ResultCard({ expanded, result }: { expanded: boolean; result: ResultRef }) {
  const [metadata, setMetadata] = useState<ResultRef>(result);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const initialNames = Object.keys(result.outputs ?? { strain: { shape: [] } });
  const [active, setActive] = useState(initialNames[0] ?? "strain");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setReady(false);
      setError(null);
      try {
        let next: ResultRef;
        try {
          next = await fetchResultMetadata(result.result_id);
        } catch {
          next = await recoverResultMetadata(result);
        }
        if (cancelled) return;
        setMetadata(next);
        const names = Object.keys(next.outputs ?? { strain: { shape: [] } });
        setActive((current) => (names.includes(current) ? current : names[0] ?? "strain"));
        setReady(true);
      } catch (err) {
        if (!cancelled) {
          const detail = err instanceof Error ? err.message : String(err);
          setError(`结果文件存在但索引丢失，请重新运行或检查后端数据目录。${detail}`);
        }
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [result]);

  const outputNames = Object.keys(metadata.outputs ?? { strain: { shape: [] } });

  return (
    <article className="artifact-card">
      <header className="artifact-header">
        <div>
          <h3>{metadata.result_key || "计算结果"}</h3>
          <span>{metadata.kind === "bnn" ? "BNN 结果" : "矩阵结果"}</span>
        </div>
        <a className="icon-button" href={downloadUrl(metadata.result_id)} title="下载 .mat 文件">
          <Download size={17} />
        </a>
      </header>
      {error && <p className="error-text">{error}</p>}
      {ready && outputNames.length > 1 && (
        <div className="tabs">
          {outputNames.map((name) => (
            <button
              className={active === name ? "active" : ""}
              key={name}
              type="button"
              onClick={() => setActive(name)}
            >
              {name}
            </button>
          ))}
        </div>
      )}
      {ready && <HeatmapViewer expanded={expanded} resultId={metadata.result_id} name={active} />}
    </article>
  );
}
