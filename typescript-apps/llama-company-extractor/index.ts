import { z } from "zod"; //  import zod for JSON schema validation

// ─── Schema ──────────────────────────────────────────────────────────────────
// Define the schema for a company object
const companySchema = z.object({
  companyName: z.string(), //  string field
  activity: z.string(), //  string field
  location: z.string().nullable(), //  string field that can be null
  sector: z.string(), //  string field
});

type Company = z.infer<typeof companySchema>; //  infer the type from the schema

// ─── JSON repair ─────────────────────────────────────────────────────────────
// The model may wrap the JSON in prose ("Here is the JSON: {...}").
// This strips everything before the first { and after the last }.

function extractJson(raw: string): string { // function to extract JSON from model output
  const start = raw.indexOf("{"); // find the first {
  const end = raw.lastIndexOf("}"); // find the last }

  if (start === -1 || end === -1) { // if no { or } found
    throw new Error(`No JSON object found in model output:\n${raw}`);
  }

  return raw.slice(start, end + 1); // return the JSON object
}

// ─── LLM call ────────────────────────────────────────────────────────────────

const LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"; //  llama server API endpoint

const SYSTEM_PROMPT = `You are a data-extraction assistant. // instruction to the model

The user will give you raw text about a company.
Return ONLY a valid JSON object with exactly these four fields:

{
  "companyName": "string",
  "activity": "string describing what the company does",
  "location": "city or country string, or null if unknown",
  "sector": "string, e.g. Food & Beverage, Software, Finance, ..."
}

No markdown fences. No explanation. No extra fields. Pure JSON only.`;

async function callLlama(text: string): Promise<string> { // function to call llama server
  const res = await fetch(LLAMA_URL, {
    method: "POST", //  POST method
    headers: { "Content-Type": "application/json" }, //  header for JSON content
    body: JSON.stringify({
      model: "local", // llama-server ignores this but the field is required
      temperature: 0.1, //  temperature for randomness
      messages: [
        { role: "system", content: SYSTEM_PROMPT },//  system prompt to the model
        { role: "user", content: text }, // user input
      ],
    }),
  });

  if (!res.ok) {// check if the response is ok,  if not, throw an error
    const body = await res.text(); // get the response body
    throw new Error(`llama-server error ${res.status}: ${body}`); // throw an error with the response body
  }

  const data = (await res.json()) as { // parse the response body
    choices: { message: { content: string } }[]; // extract the content from the response body
  };

  return data.choices[0].message.content; // return the content
}

// ─── Extract with retry ───────────────────────────────────────────────────────
// Function to extract company information from text with retry logic
async function extractCompany(
  text: string,// text to extract company information from
  maxRetries = 2// maximum number of retries
): Promise<Company> {// return type is Company
  let lastError: unknown;// variable to store the last error

  for (let attempt = 1; attempt <= maxRetries + 1; attempt++) { // retry loop to handle failed attempts
    console.log(`\n─── Attempt ${attempt} ────────────────────────────────`); // log the attempt number

    const raw = await callLlama(text); // call llama server

    console.log("RAW MODEL OUTPUT:"); // log the raw model output
    console.log(raw);

    try {
      const jsonString = extractJson(raw); // extract the JSON object from the raw output
      const parsed = JSON.parse(jsonString) as unknown; // parse the JSON object
      const company = companySchema.parse(parsed); // validate the parsed JSON object
      return company; // return the valid company object
    } catch (err) { // catch block to handle failed attempts
      lastError = err; // store the last error
      console.error(`\n⚠  Parse/validation failed on attempt ${attempt}:`);// log the error message
      console.error(err instanceof Error ? err.message : err); // log the error message

      if (attempt <= maxRetries) { // check if the number of attempts is less than the maximum number of retries
        console.log("Retrying…"); // log that the function is retrying
      }
    }
  }

  throw new Error(
    `Failed to extract a valid Company after ${maxRetries + 1} attempts.\nLast error: ${lastError}`// throw an error if the number of attempts is exceeded
  );
}

// ─── Test inputs ─────────────────────────────────────────────────────────────

const testCases: { label: string; text: string }[] = [
  {
    label: "Belgian brewery",
    text: `
      Brouwerij Omer Vander Ghinste is a Belgian family brewery located in
      Bellegem, near Kortrijk, in the West Flanders province of Belgium.
      The company has been brewing beer since 1892 and produces brands such as
      OMER. Traditional Blond, Tripel LeFort and Ypra.
    `,
  },
  {
    label: "SaaS company",
    text: `
      Datadog is a cloud-based monitoring and analytics platform for developers,
      IT operations teams, and business users in the cloud age. Headquartered in
      New York City, it enables organisations to monitor their entire technology
      stack.
    `,
  },
  {
    label: "Vague / location-less",
    text: `
      NovaTech builds AI-powered automation tools for supply-chain optimisation.
      The company raised a €12 million Series A in 2024. No public headquarters
      address is listed.
    `,
  },
];

// ─── Main ─────────────────────────────────────────────────────────────────────

for (const { label, text } of testCases) {// iterate over the test cases
  console.log(`\n${"═".repeat(60)}`); // log a separator line
  console.log(`TEST: ${label}`); // log the test case label
  console.log("═".repeat(60)); // log a separator line

  try {
    const company = await extractCompany(text); // extract the company information from the text
    console.log("\n✅ VALIDATED RESULT:"); // log the valid company object
    console.log(JSON.stringify(company, null, 2)); // log the valid company object
  } catch (err) { // catch block to handle failed attempts
    console.error("\n❌ EXTRACTION FAILED:"); // log the error message
    console.error(err instanceof Error ? err.message : err); // log the error message
  }
}
