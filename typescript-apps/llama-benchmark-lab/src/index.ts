import { benchmarks } from './benchmarks.js'; // Import the benchmarks from the benchmarks.ts file
import { saveResult, type BenchmarkResult } from './reporter.js'; // Import the saveResult function from the reporter.ts file

const LLAMA_URL = 'http://127.0.0.1:8080/v1/chat/completions'; // Llama-server API endpoint
const MODELS_URL = 'http://127.0.0.1:8080/v1/models'; // Llama-server API endpoint

async function getActiveModel(): Promise<string> {
  try {
    const res = await fetch(MODELS_URL); // Fetches the list of available models from the llama-server
    if (!res.ok) return 'unknown_model'; // If the response is not ok, return 'unknown_model'
    const data = await res.json() as { data: { id: string }[] }; // Parses the JSON response from the llama-server
    return data.data?.[0]?.id || 'unknown_model'; // Returns the ID of the first model in the list
  } catch {
    return 'unknown_model'; // Returns 'unknown_model' if the request fails
  }
}

async function runBenchmark(model: string, benchmark: typeof benchmarks[0]) {// Function to run a benchmark
  console.log(`\nRunning benchmark: "${benchmark.name}"...`);// Logs the name of the benchmark being run
  
  const res = await fetch(LLAMA_URL, {
    method: 'POST', // POST method
    headers: { 'Content-Type': 'application/json' }, // Header for JSON content
    body: JSON.stringify({
      model: 'local', // Model to use
      messages: [{ role: 'user', content: benchmark.prompt }], // User message
      temperature: 0.0, // Deterministic
      max_tokens: 500, // Maximum number of tokens to generate
      stream: true, // Stream the response
    }),
  });

  if (!res.ok) { // Checks if the response is ok
    throw new Error(`llama-server error: ${res.status}`); // Throws an error if the response is not ok
  }

  if (!res.body) throw new Error("No response body"); // Throws an error if there is no response body

  const reader = res.body.getReader(); // Reads the response body
  const decoder = new TextDecoder(); // Decodes the response body
  let buffer = ''; // Buffer to store incomplete JSON chunks
  let timings: any = null; // Object to store timings

  while (true) {
    const { value, done } = await reader.read(); // Reads the response body
    if (done) break; // Breaks the loop if the response is done

    buffer += decoder.decode(value, { stream: true }); // Decodes the response body
    const lines = buffer.split('\n'); // Splits the response body into lines
    buffer = lines.pop() ?? ''; // Removes the last line from the buffer

    for (const line of lines) { // Iterates over the lines
      if (!line.startsWith('data: ')) continue; // Checks if the line starts with 'data: '
      const dataStr = line.slice(6); // Removes the 'data: ' prefix from the line
      if (dataStr === '[DONE]') continue; // Checks if the line is '[DONE]'

      try {
        const json = JSON.parse(dataStr); // Parses the JSON response from the llama-server
        if (json.timings) {
          timings = json.timings;// Stores the timings
        }
      } catch {
        // ignore incomplete JSON chunks
      }
    }
  }

  if (!timings) {
    console.warn("Warning: No telemetry timings found in stream. Make sure llama-server supports it.");
    return;
  }

  const promptMs = timings.prompt_ms; // Time taken to process the prompt
  const promptTokens = timings.prompt_n; // Number of tokens processed in the prompt
  const promptTPS = promptTokens / (promptMs / 1000); // Tokens per second for prompt processing

  const genMs = timings.predicted_ms; // Time taken to generate the response
  const genTokens = timings.predicted_n; // Number of tokens generated
  const genTPS = genTokens / (genMs / 1000); // Tokens per second for response generation

  console.log(`   Prompt Eval : ${promptTPS.toFixed(2)} tokens/sec (${promptTokens} tokens in ${promptMs.toFixed(0)}ms)`); // Logs the prompt evaluation results
  console.log(`   Generation  : ${genTPS.toFixed(2)} tokens/sec (${genTokens} tokens in ${genMs.toFixed(0)}ms)`); // Logs the generation results

  const result: BenchmarkResult = { // Creates an object to store the benchmark results
    timestamp: new Date().toISOString(), // Timestamp of when the benchmark was run
    model,
    benchmarkName: benchmark.name, // Name of the benchmark
    promptTokens, // Number of tokens processed in the prompt
    promptMs, // Time taken to process the prompt
    promptTPS, // Tokens per second for prompt processing
    genTokens, // Number of tokens generated
    genMs, // Time taken to generate the response
    genTPS, // Tokens per second for response generation
  };

  saveResult(result); // Saves the benchmark results
}

async function main() {
  console.log('Detecting active llama.cpp model...'); // Logs that the active model is being detected
  const model = await getActiveModel(); // Gets the active model
  console.log(`Active model: ${model}`); // Logs the active model

  for (const benchmark of benchmarks) { // Iterates over the benchmarks
    try {
      await runBenchmark(model, benchmark);
    } catch (err) { // Catches errors
      console.error(`Benchmark failed:`, err); // Logs the error
    }
  }

  console.log(`\nAll benchmarks completed!`); // Logs that all benchmarks have completed
}

main().catch(console.error); // Runs the main function and catches any errors