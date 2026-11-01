# Proposal: a chat request kind

Status: proposal for review, 2026-11-01. Nothing here is merged or ruled.

## Why

A text request is one `prompt` string. A caller that wants standing instructions, earlier turns and a machine-readable answer has to pack everything into that string and parse the result itself. The first such caller is Table Kit (sessions of several participants, each with a persona and a history). Post Kit and any other tool that asks a model for JSON have the same need. This belongs in Inference Kit because it already knows which provider and model can do what.

## What it adds

* `RequestKind::Chat` (`"chat"`), and `Request::Chat(ChatParams)`.
* `ChatParams`: `model_id`, an optional `system` prompt, `messages` (a list of `ChatMessage { role: User | Assistant, content }`), `temperature`, `max_tokens`, an optional `response_schema` (JSON Schema), `session_id`, `request_id`.
* The system prompt is a field, not a message role, so a message list can never forge it.
* The answer is `Output::Text`, unchanged. A chat is text out.

## Rules

1. **Shape is checked before routing.** At least one message, the last from the user, no blank messages, temperature in [0, 2], `max_tokens` at least 1.
2. **Size limits** (proposal figures, in `limits`): 256 messages, 120,000 characters in total (system, messages and schema), schema at most 16 KiB. The text limit of 4,000 characters is too small for a persona plus history, hence a separate figure. Oversize is `RequestTooLarge`, as for every other kind.
3. **A schema is never run unconstrained.** A request with `response_schema` implies `needs.structured_outputs`, set by the router from the request, not left to the caller to remember. A model whose card does not say `structured_outputs: yes` (unknown means no) is not chosen; with none left the call fails with the usual unavailable / unsupported error and no provider is called.
4. **The kit does not promise the schema was met.** A refusal or a length stop can still return text that is not JSON. The caller parses and checks.
5. **Same egress and spend rules as text.** The label, the policy, the per-hop decision, the ledger reservation and settlement, and the audit record (no content) are the existing ones. Nothing is kept.
6. **Providers opt in.** A provider serves chat only if it lists `Chat` in its kinds. In this draft only the Anthropic adapter does.

## What this draft does

* `modality.rs`: the kind, `ChatRole`, `ChatMessage`, `ChatParams`, validation, limits, and the `Request` methods.
* `router.rs`: token estimate for chat; the schema implies the structured-output need.
* `routed.rs` (the older two-tier router): classifies a chat by its newest message; it has no chat backend.
* `providers/anthropic.rs`: sends `system` and the message list to the Messages API; declares `Chat`.
* Tests for validation, routing (text-only providers skipped, schema needs a `yes` card), and the Anthropic wire shape and billing, all with mocks.

## What it deliberately does not do

* **Structured output on Anthropic is not implemented.** The wire shape has not been checked against a live key, and sending the schema into the prompt would be a weaker guarantee than the rule above promises. The adapter refuses a schema request before reserving or sending anything. Until it is verified, cards from this adapter should not report `structured_outputs: yes`, and Post Kit / Table Kit keep their own check-and-retry loop.
* Gemini and OpenAI-compatible adapters do not declare `Chat` yet. Both have chat endpoints; each is a small follow-up.
* No streaming, no tools, no images in a chat.
* `prompt_text()` for a chat returns the newest message only. The egress label is set by the code that read the content, as for text; a classifier that reads one string will not see the system prompt.

## Questions for review

1. Is a new kind the right shape, or should `InferenceParams` grow optional `system` / `messages` fields? A kind lets a provider say what it serves and keeps the frozen text ABI alone; the cost is one more kind for every consumer to match.
2. Are the size figures right?
3. Should a chat carry a `purpose`-like hint for the egress policy, or is the label enough?
4. Should multi-turn state ever be kept by the kit (a session)? This draft says no: the caller owns history, which fits the invariant against keeping plaintext prompts.
