"""Send the local `prompt` file to OpenRouter and print 1-5 label probabilities."""

from __future__ import annotations

import argparse
from pathlib import Path

from openrouter_client import OpenRouterClient


LABEL_TOKENS = ("1", "2", "3", "4", "5")
DEFAULT_SYSTEM_PROMPT = (
    "You revise toxicity classifier outputs using monitor witness evidence. "
    "Follow the requested output format exactly."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-off OpenRouter check for a prompt stored in a file."
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path(__file__).with_name("prompt"),
        help="Prompt file to send. Default: data/text/prompt.",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt sent with the user prompt.",
    )
    parser.add_argument(
        "--preserve-final-newline",
        action="store_true",
        help=(
            "Send the prompt file exactly as stored. By default one terminal "
            "line ending is stripped so editor-added final newlines do not "
            "change the request."
        ),
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8")
    if not args.preserve_final_newline:
        prompt = strip_one_final_line_ending(prompt)
    client = OpenRouterClient(
        max_workers=1,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    result = client.judge_prompts(
        [prompt],
        label_tokens=LABEL_TOKENS,
        system_prompt=args.system_prompt,
    )[0]

    print(f"prompt_file: {args.prompt_file}")
    print(f"answer: {result.answer!r}")
    print(f"first_token: {result.first_token!r}")
    print(f"finish_reason: {result.finish_reason}")
    print(f"model: {result.model}")
    if result.error:
        print(f"error: {result.error}")
    print("probabilities:")
    for label in LABEL_TOKENS:
        prob = result.label_probs.get(label)
        logprob = result.label_logprobs.get(label)
        source = result.label_logprob_sources.get(label)
        prob_s = "n/a" if prob is None else f"{prob:.8f}"
        logprob_s = "n/a" if logprob is None else f"{logprob:.8f}"
        print(f"  {label}: prob={prob_s} logprob={logprob_s} source={source}")


def strip_one_final_line_ending(text: str) -> str:
    if text.endswith("\r\n"):
        return text[:-2]
    if text.endswith("\n") or text.endswith("\r"):
        return text[:-1]
    return text


if __name__ == "__main__":
    main()
