// llm.ts — thin wrapper around the llama-server OpenAI-compatible HTTP API.
//
// Architecture:
//
//   application
//     ↓
//   askLocalModel()
//     ↓
//   OpenAI-compatible HTTP API
//     ↓
//   llama-server
//     ↓
//   Qwen GGUF
//     ↓
//   GGML / Metal / Apple GPU

const LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions";

export interface LlmOptions {
  temperature?: number;
  maxTokens?: number;
}

export async function askLocalModel(
  systemPrompt: string,
  userPrompt: string,
  options: LlmOptions = {}
): Promise<string> {
  const response = await fetch(LLAMA_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "local", // llama-server ignores this — it serves whatever is loaded
      temperature: options.temperature ?? 0.1,
      max_tokens: options.maxTokens ?? 512,
      // /no_think disables Qwen3's reasoning chain.
      // Without it, the answer goes to reasoning_content and content stays empty.
      messages: [
        { role: "system", content: systemPrompt + "\n/no_think" },
        { role: "user", content: userPrompt },
      ],
    }),
  });

  if (!response.ok) {
    throw new Error(
      `llama-server returned ${response.status}: ${await response.text()}`
    );
  }

  const data = await response.json();
  return data.choices[0].message.content as string;
}
