import React from 'react';
import type { Metrics } from '../App';

interface Props {
  metrics: Metrics | null;
}

export function PerformanceMetrics({ metrics }: Props) {
  if (!metrics) {
    return (
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-label">Prompt Speed</div>
          <div className="metric-value">--</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Gen Speed</div>
          <div className="metric-value">--</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Total Time</div>
          <div className="metric-value">--</div>
        </div>
      </div>
    );
  }

  // Calculated values
  const promptToks = metrics.promptTokens / (metrics.promptMs / 1000);
  const genToks = metrics.generationTokens / (metrics.generationMs / 1000);
  const totalSecs = (metrics.promptMs + metrics.generationMs) / 1000;

  return (
    <div className="metrics-grid">
      <div className="metric-card">
        <div className="metric-label">Prompt Speed</div>
        <div className="metric-value">{isNaN(promptToks) || !isFinite(promptToks) ? '0.00' : promptToks.toFixed(2)} t/s</div>
      </div>
      <div className="metric-card">
        <div className="metric-label">Gen Speed</div>
        <div className="metric-value">{isNaN(genToks) || !isFinite(genToks) ? '0.00' : genToks.toFixed(2)} t/s</div>
      </div>
      <div className="metric-card">
        <div className="metric-label">Total Time</div>
        <div className="metric-value">{totalSecs.toFixed(2)}s</div>
      </div>
    </div>
  );
}
