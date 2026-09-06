import React, { useState, useRef, useEffect } from 'react';
import { Send, Activity, Settings2, TerminalSquare } from 'lucide-react';
import { RuntimeControls } from './components/RuntimeControls';
import { PerformanceMetrics } from './components/PerformanceMetrics';
import { ChatInterface } from './components/ChatInterface';
import './index.css';

export interface GenerationConfig {
  temperature: number;
  top_p: number;
  max_tokens: number;
}

export interface Metrics {
  promptTokens: number;
  generationTokens: number;
  promptMs: number;
  generationMs: number;
}

function App() {
  const [config, setConfig] = useState<GenerationConfig>({
    temperature: 0.7,
    top_p: 0.9,
    max_tokens: 200,
  });

  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [model, setModel] = useState<string>('Loading...');

  // Fetch current model on load
  useEffect(() => {
    fetch('http://127.0.0.1:8080/v1/models')
      .then((res) => res.json())
      .then((data) => {
        if (data.data && data.data.length > 0) {
          setModel(data.data[0].id);
        } else {
          setModel('Unknown Model');
        }
      })
      .catch(() => setModel('Disconnected (Server offline)'));
  }, []);

  return (
    <div className="dashboard-container">
      {/* Sidebar Controls */}
      <aside className="sidebar">
        <div className="panel">
          <h2><Settings2 size={20} /> Settings</h2>
          <div className="model-badge">{model}</div>
          <RuntimeControls config={config} onChange={setConfig} />
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        <div className="panel" style={{ padding: '1rem 1.5rem' }}>
          <h2><Activity size={20} /> Runtime Performance</h2>
          <PerformanceMetrics metrics={metrics} />
        </div>

        <div className="panel chat-panel">
          <ChatInterface 
            config={config} 
            onMetricsUpdate={setMetrics} 
          />
        </div>
      </main>
    </div>
  );
}

export default App;
