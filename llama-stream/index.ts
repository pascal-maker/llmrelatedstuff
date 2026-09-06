const LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions";

console.log("Connecting to llama-server...");// llama server running on port 8080

const response = await fetch(LLAMA_URL, { // llama server ip and port can be changed here
  method: "POST",
  headers: {// this part might need to change as per the llama server documentation
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    model: "Qwen/Qwen2-0.5B-Instruct-GGUF:Q8_0", // model name can be changed here
    messages: [
      {
        role: "user",
        content: "Explain in simple terms how llama.cpp generates one token.", // prompt can be changed here
      },
    ],
    temperature: 0.7, // temperature can be changed here
    top_p: 0.9, // top_p can be changed here
    max_tokens: 100, // max_tokens can be changed here
    stream: true,
  }), // options can be changed here 
});

if (!response.ok) { // checking for errors in the request
  throw new Error(
    `Request failed: ${response.status} ${await response.text()}`
  );
}

if (!response.body) { // checking for response body
  throw new Error("No response body");
}

const reader = response.body.getReader();// reader to read the response
const decoder = new TextDecoder();// decoder to decode the response
let buffer = "";// buffer to store the response

process.stdout.write("\n"); // this writes a newline character to the console

while (true) { // this loop will run until the response is done
  const { value, done } = await reader.read();

  if (done) { // checking if the response is done
    break;
  }

  buffer += decoder.decode(value, { stream: true }); // decoding the response
  const lines = buffer.split("\n"); // splitting the response into lines
  buffer = lines.pop() ?? ""; // removing the last line from the buffer

  for (const line of lines) { // looping through the lines
    if (!line.startsWith("data: ")) { // checking if the line starts with "data: "
      continue;
    }

    const data = line.slice(6); // removing "data: " from the line

    if (data === "[DONE]") { // checking if the line is "[DONE]"
      continue;
    }

    try {
      const json = JSON.parse(data); // parsing the data
      const delta = json.choices?.[0]?.delta; // getting the delta
      const token = delta?.content || delta?.reasoning_content; // getting the token

      if (token) { // checking if the token is not empty
        process.stdout.write(token); // writing the token to the console
      }
    } catch {
      // ignore incomplete SSE chunks
    }
  }
}

console.log("\n");
