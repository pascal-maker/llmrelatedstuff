// index.ts — two-step local LLM agent:
//
//   unstructured text
//     ↓
//   extractCompany()   ← LLM step 1
//     ↓
//   Zod (Company)      ← trust boundary
//     ↓
//   scoreLead()        ← LLM step 2
//     ↓
//   Zod (LeadScore)    ← trust boundary
//     ↓
//   application output
//
// Each step has a retry/repair loop: if the model returns broken JSON,
// we send the validation error back to the model and ask it to fix it.
// This is the simplest possible self-healing agent pattern.

import { askLocalModel } from "./llm.js"; //askOpenAI
import {
  companySchema, // Company 
  leadScoreSchema, // LeadScore
  type Company,
  type LeadScore,
} from "./schemas.js";

// ─── JSON helpers ─────────────────────────────────────────────────────────────

/** Strip markdown fences and extract the first {...} block. */
function extractJson(raw: string): string {
  // Strip ` ```json ` or ` ``` ` fences
  const stripped = raw
    .replace(/^```(?:json)?\s*/im, "")
    .replace(/\s*```$/im, "")
    .trim();

  const start = stripped.indexOf("{");
  const end = stripped.lastIndexOf("}");
  if (start === -1 || end === -1) {
    throw new Error(`No JSON object found in model output:\n${raw}`);
  }
  return stripped.slice(start, end + 1);
}

// ─── Retry/repair loop ────────────────────────────────────────────────────────
//
// Pattern:
//   LLM output
//     ↓
//   JSON.parse()
//     ↓  ← if invalid ──────────────────────────────────────────────────┐
//   Zod validation                                                       │
//     ↓  ← if invalid ──────────────────────────────────────────────────┤
//   trusted object                                               send error back
//                                                                to model → retry

async function withRetry<T>(
  label: string,
  systemPrompt: string,
  firstUserPrompt: string,
  validate: (raw: string) => T,
  maxRetries = 2
): Promise<T> {
  let userPrompt = firstUserPrompt;
  let lastError: unknown;

  for (let attempt = 1; attempt <= maxRetries + 1; attempt++) {
    if (attempt > 1) {
      console.log(`\n  ↻ Retry ${attempt - 1}/${maxRetries} (repairing output)…`);
    }

    const raw = await askLocalModel(systemPrompt, userPrompt);
    console.log(`\n[${label}] raw output:\n${raw}`);

    try {
      return validate(raw);
    } catch (err) {
      lastError = err;
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`\n  ⚠  ${label} parse/validation failed: ${msg}`);

      // Feed the error back to the model so it can self-correct
      userPrompt = `Your previous output caused this error:\n${msg}\n\nPlease return ONLY valid JSON that fixes the error. No prose, no fences.`;
    }
  }

  throw new Error(
    `${label} failed after ${maxRetries + 1} attempts. Last error: ${lastError}`
  );
}

// ─── Step 1: Extract company facts ───────────────────────────────────────────

const EXTRACTION_PROMPT = `You are a structured data extraction agent.

The user will give you raw text about a company.
Return ONLY a valid JSON object — no prose, no markdown, no fences.

Required schema:
{
  "companyName": "string",
  "activity": "string",
  "sector": "string",
  "locations": ["string"],
  "employeeEstimate": number | null,
  "ownsIndustrialProperty": boolean | null,
  "evidence": ["string — short fact from the input"]
}

Rules:
- Never invent facts not present in the text.
- If a field is unknown, use null (for numbers/booleans) or [] (for arrays).
- evidence must contain only statements directly supported by the input.`;

async function extractCompany(text: string): Promise<Company> {
  return withRetry(
    "EXTRACTION",
    EXTRACTION_PROMPT,
    text,
    (raw) => {
      const json = JSON.parse(extractJson(raw)) as unknown;
      return companySchema.parse(json);
    }
  );
}

// ─── Step 2: Score commercial fit ────────────────────────────────────────────

const SCORING_PROMPT = `You are a commercial real estate lead scoring agent.

We look for companies that may need industrial, warehouse,
logistics, or production property in Belgium.

Score the company from 0 to 100.

Score higher when:
- manufacturing or production activity is present
- the company operates warehouses or storage
- logistics or distribution operations exist
- the company has multiple physical locations
- there are signs of growth or expansion

Return ONLY a valid JSON object — no prose, no markdown, no fences.

Required schema:
{
  "score": number (0–100),
  "verdict": "strong_fit" | "possible_fit" | "weak_fit",
  "reasons": ["string"],
  "nextAction": "string"
}`;

async function scoreLead(company: Company): Promise<LeadScore> {
  return withRetry(
    "SCORING",
    SCORING_PROMPT,
    JSON.stringify(company, null, 2),
    (raw) => {
      const json = JSON.parse(extractJson(raw)) as unknown;
      return leadScoreSchema.parse(json);
    }
  );
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function researchCompany(text: string): Promise<void> {
  console.log("\n" + "═".repeat(60));
  console.log("STEP 1 — Company extraction");
  console.log("═".repeat(60));

  const company = await extractCompany(text);

  console.log("\n✅ VALIDATED COMPANY:");
  console.log(JSON.stringify(company, null, 2));

  console.log("\n" + "═".repeat(60));
  console.log("STEP 2 — Lead scoring");
  console.log("═".repeat(60));

  const lead = await scoreLead(company);

  console.log("\n✅ LEAD RESULT:");
  console.log(JSON.stringify({
    company: company.companyName,
    score: lead.score,
    verdict: lead.verdict,
    reasons: lead.reasons,
    nextAction: lead.nextAction,
  }, null, 2));
}

await researchCompany(`
Brouwerij Omer Vander Ghinste is a Belgian family-owned brewery
located in Bellegem near Kortrijk.

The brewery produces several Belgian beers including OMER. Traditional Blond,
Tripel LeFort and Ypra. Beer production, bottling and storage take place at
the brewery site in Bellegem.

The company distributes its products throughout Belgium and exports beer
internationally.
`);
