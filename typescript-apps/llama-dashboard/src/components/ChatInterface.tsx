import React, { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';
import type { GenerationConfig, Metrics } from '../App';

interface Props {
  config: GenerationConfig;
  onMetricsUpdate: (metrics: Metrics) => void;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export function ChatInterface({ config, onMetricsUpdate }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isGenerating) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage, { role: 'assistant', content: '' }]);
    setInput('');
    setIsGenerating(true);

    try {
      const response = await fetch('http://127.0.0.1:8080/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: 'local', // Ignored by llama-server
          messages: [...messages, userMessage],
          temperature: config.temperature,
          top_p: config.top_p,
          max_tokens: config.max_tokens,
          stream: true,
        }),
      });

      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          
          const dataStr = line.slice(6);
          if (dataStr === '[DONE]') continue;

          try {
            const json = JSON.parse(dataStr);
            
            // Handle telemetry from llama-server (usually comes in last chunk or DONE chunk if configured, 
            // but OpenAI format puts it in choices[0].finish_reason or an extra 'timings' object sometimes)
            if (json.timings) {
              onMetricsUpdate({
                promptTokens: json.timings.prompt_n,
                promptMs: json.timings.prompt_ms,
                generationTokens: json.timings.predicted_n,
                generationMs: json.timings.predicted_ms
              });
            }

            const delta = json.choices?.[0]?.delta;
            if (!delta) continue;
            
            // Handle both standard content and Qwen reasoning_content
            const token = delta.content || delta.reasoning_content;

            if (token) {
              setMessages((prev) => {
                const updated = [...prev];
                updated[updated.length - 1].content += token;
                return updated;
              });
            }
          } catch (err) {
            // ignore JSON parse errors for incomplete chunks
          }
        }
      }
    } catch (error) {
      console.error('Generation failed:', error);
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1].content += '\n\n[Error communicating with server]';
        return updated;
      });
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <>
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)', marginTop: '2rem' }}>
            Send a message to start streaming...
          </div>
        ) : (
          messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              {msg.content}
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      <form onSubmit={handleSubmit} className="chat-input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask the local model..."
          disabled={isGenerating}
        />
        <button type="submit" disabled={isGenerating || !input.trim()}>
          <Send size={18} className={isGenerating ? 'generating' : ''} />
        </button>
      </form>
    </>
  );
}
