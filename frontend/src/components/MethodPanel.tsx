import { SlidersHorizontal } from "lucide-react";
import type { MethodSettings } from "../types/api";

type Props = {
  settings: MethodSettings;
  onChange: (settings: MethodSettings) => void;
};

export function MethodPanel({ settings, onChange }: Props) {
  function patch(next: Partial<MethodSettings>) {
    onChange({ ...settings, ...next });
  }

  function patchPhysical(next: Partial<MethodSettings["physical"]>) {
    onChange({ ...settings, physical: { ...settings.physical, ...next } });
  }

  return (
    <details className="advanced-settings">
      <summary>
        <SlidersHorizontal size={16} />
        <span>高级设置</span>
      </summary>
      <div className="advanced-body">
        <fieldset>
          <legend>显示设置</legend>
          <label className="check-row">
            <input
              type="checkbox"
              checked={settings.visualizationEnabled}
              onChange={(event) => patch({ visualizationEnabled: event.target.checked })}
            />
            <span>可视化判断</span>
          </label>
          <label className="check-row">
            <input
              type="checkbox"
              checked={settings.showThinking}
              onChange={(event) => patch({ showThinking: event.target.checked })}
            />
            <span>显示深度思考</span>
          </label>
        </fieldset>

        <fieldset>
          <legend>物理参数</legend>
          <div className="number-grid">
            <label>
              波长
              <input
                type="number"
                step="1e-9"
                value={settings.physical.wavelength}
                onChange={(event) => patchPhysical({ wavelength: Number(event.target.value) })}
              />
            </label>
            <label>
              带宽
              <input
                type="number"
                step="1e-9"
                value={settings.physical.bandwidth}
                onChange={(event) => patchPhysical({ bandwidth: Number(event.target.value) })}
              />
            </label>
            <label>
              折射率
              <input
                type="number"
                step="0.01"
                value={settings.physical.refractiveIndex}
                onChange={(event) => patchPhysical({ refractiveIndex: Number(event.target.value) })}
              />
            </label>
          </div>
        </fieldset>

        <fieldset>
          <legend>应变计算方法</legend>
          <label className="check-row">
            <input
              type="checkbox"
              checked={settings.vector}
              onChange={(event) => patch({ vector: event.target.checked })}
            />
            <span>矢量法</span>
          </label>
          <div className="number-grid">
            <label>
              Nx
              <input
                type="number"
                min={1}
                value={settings.Nx}
                onChange={(event) => patch({ Nx: Number(event.target.value) })}
              />
            </label>
            <label>
              Nz
              <input
                type="number"
                min={1}
                value={settings.Nz}
                onChange={(event) => patch({ Nz: Number(event.target.value) })}
              />
            </label>
            <label>
              g
              <input
                type="number"
                min={1}
                value={settings.g}
                onChange={(event) => patch({ g: Number(event.target.value) })}
              />
            </label>
          </div>
          <label className="check-row">
            <input
              type="checkbox"
              checked={settings.cnn}
              onChange={(event) => patch({ cnn: event.target.checked })}
            />
            <span>CNN</span>
          </label>
          <label className="check-row">
            <input
              type="checkbox"
              checked={settings.bnn}
              onChange={(event) => patch({ bnn: event.target.checked })}
            />
            <span>BNN</span>
          </label>
          <label className="wide-number">
            MC_test
            <input
              type="number"
              min={1}
              max={200}
              value={settings.MC_test}
              onChange={(event) => patch({ MC_test: Number(event.target.value) })}
            />
          </label>
        </fieldset>
      </div>
    </details>
  );
}
