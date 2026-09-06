import fs from 'fs';
import path from 'path';

export interface BenchmarkResult {
  timestamp: string;
  model: string;
  benchmarkName: string;
  promptTokens: number;
  promptMs: number;
  promptTPS: number;
  genTokens: number;
  genMs: number;
  genTPS: number;
}

const RESULTS_DIR = path.resolve(process.cwd(), 'results');
const CSV_FILE = path.join(RESULTS_DIR, 'benchmark_results.csv');

export function saveResult(result: BenchmarkResult) {
  if (!fs.existsSync(RESULTS_DIR)) {
    fs.mkdirSync(RESULTS_DIR, { recursive: true });
  }

  const isNewFile = !fs.existsSync(CSV_FILE);
  const header = `Timestamp,Model,Benchmark,Prompt Tokens,Prompt ms,Prompt t/s,Gen Tokens,Gen ms,Gen t/s\n`;
  
  const row = `${result.timestamp},"${result.model}","${result.benchmarkName}",${result.promptTokens},${result.promptMs.toFixed(2)},${result.promptTPS.toFixed(2)},${result.genTokens},${result.genMs.toFixed(2)},${result.genTPS.toFixed(2)}\n`;

  if (isNewFile) {
    fs.writeFileSync(CSV_FILE, header + row, 'utf8');
  } else {
    fs.appendFileSync(CSV_FILE, row, 'utf8');
  }

  console.log(`\n💾 Saved result to ${CSV_FILE}`);
}
