import { z } from "zod";

// ─── Company ──────────────────────────────────────────────────────────────────

export const companySchema = z.object({
  companyName: z.string(),
  activity: z.string(),
  sector: z.string(),
  locations: z.array(z.string()),
  employeeEstimate: z.number().nullable(),
  ownsIndustrialProperty: z.boolean().nullable(),
  evidence: z.array(z.string()),
});

export type Company = z.infer<typeof companySchema>;

// ─── LeadScore ────────────────────────────────────────────────────────────────

export const leadScoreSchema = z.object({
  score: z.number().min(0).max(100),
  verdict: z.enum(["strong_fit", "possible_fit", "weak_fit"]),
  reasons: z.array(z.string()),
  nextAction: z.string(),
});

export type LeadScore = z.infer<typeof leadScoreSchema>;
