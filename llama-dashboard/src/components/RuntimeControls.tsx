import React from 'react';
import type { GenerationConfig } from '../App';

interface Props {
  config: GenerationConfig;
  onChange: (config: GenerationConfig) => void;
}

export function RuntimeControls({ config, onChange }: Props) {
  const updateConfig = (key: keyof GenerationConfig, value: number) => {
    onChange({ ...config, [key]: value });
  };

  return (
    <div className="runtime-controls">
      <div className="control-group">
        <div className="control-header">
          <span>Temperature</span>
          <span className="control-value">{config.temperature.toFixed(2)}</span>
        </div>
        <input 
          type="range" 
          min="0" 
          max="2" 
          step="0.05"
          value={config.temperature}
          onChange={(e) => updateConfig('temperature', parseFloat(e.target.value))}
        />
      </div>

      <div className="control-group">
        <div className="control-header">
          <span>Top P</span>
          <span className="control-value">{config.top_p.toFixed(2)}</span>
        </div>
        <input 
          type="range" 
          min="0" 
          max="1" 
          step="0.05"
          value={config.top_p}
          onChange={(e) => updateConfig('top_p', parseFloat(e.target.value))}
        />
      </div>

      <div className="control-group">
        <div className="control-header">
          <span>Max Tokens</span>
          <span className="control-value">{config.max_tokens}</span>
        </div>
        <input 
          type="range" 
          min="10" 
          max="2048" 
          step="10"
          value={config.max_tokens}
          onChange={(e) => updateConfig('max_tokens', parseInt(e.target.value, 10))}
        />
      </div>
    </div>
  );
}
