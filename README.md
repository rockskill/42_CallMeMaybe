_This project has been created as part of the 42 curriculum by roiskusn._

# 42 School: CallMeMaybe

## Description

42 School Core project that used small language model (LM) (Qwen3-0.6B) to translate natural language prompts into structured function calls. The output is stored to a JSON file and each call is represented as structured JSON-record (including prompt, function name and its input parameters). The whole project is written in Python and custom LLM SDK.

## Instructions

### Installation

```bash
# Clone the repository.
git clone <repository_url> <folder_for_cloning>
cd <folder_for_cloning>

# Copy llm_sdk into the project repository root.
cp -r <path_to_sdk> .

# Install project dependencies.
make install
```

### Running

```bash
# Default path (data/input/ → data/output/)
make run

# Custom path
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

### Other commands for Make
```bash
make debug # Runs a debugger.
make lint # Runs linter (flake8 and mypy).
make lint-strict # Runs linter in a strict mode.
make clean # Cleans up caches.
```

## Algorithm

### Constrained Decoding

The algorithm generates output token-by-token and restricts which tokens are allowed at each step:

1. **Selects function**: allows only tokens that are valid continuations of known function names. The model picks the most likely token from this restricted set, progressively narrowing candidates until one function remains. The rest of the name is appended without LLM call.

2. **Generates arguments (option-based approach)**: for string arguments, complete phrases are extracted from the prompt and tokenized. Only the first token of each remaining option is allowed at each step (all options that do not start with that token are eliminated). It repeats until one option remains which is then appended in full without further LLM calls.

3. **Typed arguments**:
   - `number` / `integer`: extracted from the prompt via options, validated and converted to float;
   - `boolean`: restricted to `true` / `false` only;
   - `regex`: resolved deterministically via keyword matching.

4. **Regex arguments**: The prompt is scanned for known keywords (`vowels`, `digits`, `spaces`, etc.) and mapped to the corresponding regex pattern (`[aeiouAEIOU]`, `\d+`, `\s+`, etc.). For literal word replacements (e.g. `"cat"`), the word is extracted directly from the prompt.

### Model prompting method

The model is prompted in Qwen-native tool-calling format:
```
<|im_start|>system
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name": "fn_add_numbers", ...}
</tools>
For each function call, return a json object within <tool_call></tool_call> tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
<|im_end|>
<|im_start|>user
What is the sum of 2 and 3?
<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "fn_add_numbers", "arguments": {"a": 2, "b": 3}}
```
The format above matches Qwen3's training format and improves argument extraction quality compared to a custom JSON format.

## Resources
[1] Qwen3-0.6B official documentation.
[2] Qwen3 model tool calling format.
[3] Byte-Pair Encoding algorithm (BPE) explanation and benefits of use.
[4] Pydantic official documentation.