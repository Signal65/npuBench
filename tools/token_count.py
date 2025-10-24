import argparse
import json
import sys
from typing import Any, Dict, List, Optional


def load_text(path: Optional[str], inline: Optional[str]) -> str:
    if path:
        with open(path, "r", encoding="utf-8-sig") as f:
            return f.read()
    return inline or ""


def load_messages(path: Optional[str], inline: Optional[str]) -> List[Dict[str, Any]]:
    data = load_text(path, inline)
    if not data:
        return []
    try:
        obj = json.loads(data)
        if isinstance(obj, list):
            return obj
        raise ValueError("messages must be a JSON list")
    except Exception as e:
        raise SystemExit(f"Failed to parse messages JSON: {e}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--messages-path")
    p.add_argument("--messages-json")
    p.add_argument("--completion-path")
    p.add_argument("--completion-text")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()

    try:
        from transformers import AutoTokenizer
    except Exception as e:
        print(json.dumps({"error": f"transformers import failed: {e}"}))
        return 2

    try:
        tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, local_files_only=args.local_files_only)
    except Exception as e:
        print(json.dumps({"error": f"tokenizer load failed: {e}"}))
        return 3

    messages = load_messages(args.messages_path, args.messages_json)
    completion = load_text(args.completion_path, args.completion_text)

    # Render chat using the model's template if available
    try:
        if hasattr(tok, "apply_chat_template") and messages and isinstance(messages[0], dict) and "role" in messages[0]:
            rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_ids = tok.encode(rendered, add_special_tokens=False)
        else:
            joined = "\n\n".join(m if isinstance(m, str) else m.get("content", "") for m in messages)
            prompt_ids = tok.encode(joined, add_special_tokens=True)
        completion_ids = tok.encode(completion or "", add_special_tokens=False)
    except Exception as e:
        print(json.dumps({"error": f"tokenization failed: {e}"}))
        return 4

    print(json.dumps({
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids)
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
