import Plot from "react-plotly.js";
import { useEffect, useMemo, useState } from "react";
import { fetchHeatmap } from "../lib/api";
import type { HeatmapPayload } from "../types/api";

type Props = {
  expanded?: boolean;
  resultId: string;
  name: string;
};

function formatOne(value: number): string {
  return Number.isFinite(value) ? value.toFixed(1) : "0.0";
}

export function HeatmapViewer({ expanded = false, resultId, name }: Props) {
  const [payload, setPayload] = useState<HeatmapPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [range, setRange] = useState<[number, number] | null>(null);
  const [draftRange, setDraftRange] = useState<[string, string]>(["", ""]);

  useEffect(() => {
    let cancelled = false;
    setPayload(null);
    setError(null);
    fetchHeatmap(resultId, name)
      .then((data) => {
        if (cancelled) return;
        setPayload(data);
        setRange([data.min, data.max]);
        setDraftRange([formatOne(data.min), formatOne(data.max)]);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [resultId, name]);

  const step = useMemo(() => {
    if (!payload) return 0.1;
    const span = Math.abs(payload.max - payload.min);
    return span > 0 ? span / 200 : 0.1;
  }, [payload]);

  if (error) return <p className="error-text">{error}</p>;
  if (!payload || !range) return <div className="heatmap-loading">正在加载热力图...</div>;
  const plotHeight = expanded ? 620 : 310;

  function updateRange(next: [number, number]) {
    setRange(next);
    setDraftRange([formatOne(next[0]), formatOne(next[1])]);
  }

  function commitDraft(index: 0 | 1) {
    if (!range) return;
    const parsed = Number(draftRange[index]);
    if (!Number.isFinite(parsed)) {
      setDraftRange([formatOne(range[0]), formatOne(range[1])]);
      return;
    }
    const next: [number, number] = index === 0 ? [parsed, range[1]] : [range[0], parsed];
    updateRange(next);
  }

  return (
    <div className={`heatmap-viewer ${expanded ? "expanded" : ""}`}>
      <div className="heatmap-summary">
        <span>最小值：{formatOne(range[0])}</span>
        <span>最大值：{formatOne(range[1])}</span>
      </div>
      <div className="range-controls">
        <div className="range-control">
          <label>
            最小值
            <input
              type="text"
              inputMode="decimal"
              value={draftRange[0]}
              onBlur={() => commitDraft(0)}
              onChange={(event) => setDraftRange([event.target.value, draftRange[1]])}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  commitDraft(0);
                }
              }}
            />
          </label>
          <input
            type="range"
            min={payload.min}
            max={payload.max}
            step={step}
            value={range[0]}
            onChange={(event) => updateRange([Number(event.target.value), range[1]])}
            title="色阶最小值"
          />
        </div>
        <div className="range-control">
          <label>
            最大值
            <input
              type="text"
              inputMode="decimal"
              value={draftRange[1]}
              onBlur={() => commitDraft(1)}
              onChange={(event) => setDraftRange([draftRange[0], event.target.value])}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  commitDraft(1);
                }
              }}
            />
          </label>
          <input
            type="range"
            min={payload.min}
            max={payload.max}
            step={step}
            value={range[1]}
            onChange={(event) => updateRange([range[0], Number(event.target.value)])}
            title="色阶最大值"
          />
        </div>
      </div>
      <Plot
        data={[
          {
            z: payload.data,
            type: "heatmap",
            colorscale: "Jet",
            zmin: Math.min(range[0], range[1]),
            zmax: Math.max(range[0], range[1]),
            colorbar: { title: { text: name } },
          },
        ]}
        layout={{
          autosize: true,
          height: plotHeight,
          margin: { l: 44, r: 16, t: 8, b: 36 },
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { color: "#1f2937" },
          xaxis: { title: { text: "X" }, gridcolor: "#e5e7eb" },
          yaxis: { title: { text: "Z" }, gridcolor: "#e5e7eb", autorange: "reversed" },
        }}
        config={{ displayModeBar: false, displaylogo: false, responsive: true }}
        className="plot"
        useResizeHandler
      />
    </div>
  );
}
