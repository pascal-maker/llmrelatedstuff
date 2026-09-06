export interface Benchmark {
  name: string;
  prompt: string;
}

export const benchmarks: Benchmark[] = [
  {
    name: "Short Generation",
    prompt: "Write a haiku about a robot learning to paint.",
  },
  {
    name: "Code Generation",
    prompt: "Write a Python function to perform binary search on a sorted array, including type hints and a docstring.",
  },
  {
    name: "Context Extraction",
    prompt: `Read the following text and extract the key entities (names, organizations, locations).
    
Text:
The Acme Corporation was founded in 1948 by John Doe and Jane Smith in Chicago, Illinois. It initially started as a small manufacturing business producing anvils and rocket skates. By 1972, under the leadership of CEO Michael Johnson, the company expanded its operations to New York and Los Angeles, acquiring several smaller competitors along the way. Today, Acme Corp is a global conglomerate headquartered in Seattle, Washington.

Return the entities in a bulleted list.`,
  },
];
