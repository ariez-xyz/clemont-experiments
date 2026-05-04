"""Small OpenRouter helper for text experiment preprocessing.

The public methods accept batches. Embeddings are sent as true batched API
requests; chat completions are parallelized client-side because the chat API is
one completion request per prompt.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_CHAT_MODEL = "google/gemma-4-26b-a4b-it"
DEFAULT_EMBEDDING_MODEL = "perplexity/pplx-embed-v1-4b"
DEFAULT_CHAT_PROVIDER = {
    "only": ["nextbit"],
    "allow_fallbacks": False,
    "require_parameters": True,
}
DEFAULT_REASONING = {"effort": "none"}
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 64
DEFAULT_MAX_TOKENS = 1
DEFAULT_TOP_LOGPROBS = 20
DEFAULT_MAX_WORKERS = 8
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_APP_NAME = "clemont-experiments"
DEFAULT_EMBEDDING_BATCH_SIZE = 64
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class JudgeResult:
    """Parsed first-token judgement with top-logprob scores for label tokens."""

    answer: str
    first_token: Optional[str]
    first_token_logprob: Optional[float]
    top_logprobs: tuple[dict[str, Any], ...]
    label_logprobs: dict[str, Optional[float]]
    label_logprob_sources: dict[str, str]
    label_logprob_floor: Optional[float]
    label_probs: dict[str, Optional[float]]
    finish_reason: Optional[str]
    model: Optional[str]
    response_id: Optional[str]
    error: Optional[str] = None

    @classmethod
    def errored(
        cls,
        error: BaseException,
        *,
        label_tokens: Sequence[str] = ("0", "1"),
    ) -> "JudgeResult":
        label_logprobs = {label: None for label in label_tokens}
        return cls(
            answer="",
            first_token=None,
            first_token_logprob=None,
            top_logprobs=(),
            label_logprobs=label_logprobs,
            label_logprob_sources={label: "error" for label in label_tokens},
            label_logprob_floor=None,
            label_probs={label: None for label in label_tokens},
            finish_reason="error",
            model=None,
            response_id=None,
            error=f"{error.__class__.__name__}: {error}",
        )


class OpenRouterClient:
    """OpenRouter API wrapper for LLM-judge preprocessing jobs."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        chat_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_logprobs: Optional[int] = None,
        max_workers: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        app_name: Optional[str] = None,
        site_url: Optional[str] = None,
        provider: Optional[Mapping[str, Any]] = None,
        chat_provider: Optional[Mapping[str, Any]] = None,
        embedding_provider: Optional[Mapping[str, Any]] = None,
        reasoning: Optional[Mapping[str, Any]] = None,
        env_path: Optional[Path] = None,
    ) -> None:
        load_env_file(env_path)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required")

        chat_model = chat_model or DEFAULT_CHAT_MODEL
        embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
        temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
        top_p = DEFAULT_TOP_P if top_p is None else top_p
        max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
        top_logprobs = DEFAULT_TOP_LOGPROBS if top_logprobs is None else top_logprobs
        max_workers = DEFAULT_MAX_WORKERS if max_workers is None else max_workers
        timeout_seconds = (
            DEFAULT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        max_retries = DEFAULT_MAX_RETRIES if max_retries is None else max_retries
        app_name = app_name or DEFAULT_APP_NAME
        reasoning = dict(DEFAULT_REASONING if reasoning is None else reasoning)

        if top_logprobs < 0 or top_logprobs > 20:
            raise ValueError("top_logprobs must be between 0 and 20")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")

        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.top_logprobs = top_logprobs
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.app_name = app_name
        self.site_url = site_url
        self.chat_provider = (
            dict(
                chat_provider
                if chat_provider is not None
                else (provider if provider is not None else DEFAULT_CHAT_PROVIDER)
            )
        )
        self.embedding_provider = (
            dict(embedding_provider or provider) if (embedding_provider or provider) else None
        )
        self.reasoning = reasoning

    def judge_prompts(
        self,
        prompts: Sequence[str],
        *,
        label_tokens: Sequence[str] = ("0", "1"),
        system_prompt: Optional[str] = None,
        continue_on_error: bool = True,
    ) -> list[JudgeResult]:
        """Judge prompts in parallel, preserving input order."""

        results: list[Optional[JudgeResult]] = [None] * len(prompts)
        if not prompts:
            return []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._judge_one,
                    prompt,
                    label_tokens=tuple(label_tokens),
                    system_prompt=system_prompt,
                ): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    results[idx] = JudgeResult.errored(
                        exc,
                        label_tokens=tuple(label_tokens),
                    )

        return [result for result in results if result is not None]

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: Optional[int] = None,
    ) -> list[list[float]]:
        """Generate embeddings in API batches, preserving input order."""

        batch_size = DEFAULT_EMBEDDING_BATCH_SIZE if batch_size is None else batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            payload: dict[str, Any] = {
                "model": self.embedding_model,
                "input": chunk,
                "encoding_format": "float",
            }
            if self.embedding_provider is not None:
                payload["provider"] = self.embedding_provider

            response = self._post_json("/embeddings", payload)
            data = sorted(response["data"], key=lambda item: item.get("index", 0))
            embeddings.extend([item["embedding"] for item in data])

        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"embedding count mismatch: got {len(embeddings)} for {len(texts)} texts"
            )
        return embeddings

    def write_dataset_output(
        self,
        *,
        json_path: Optional[Path] = None,
        csv_path: Optional[Path] = None,
        output_prefix: Optional[Path] = None,
        class_count: Optional[int] = None,
        sample_size: Optional[int] = None,
        frame: pd.DataFrame,
        metadata: Mapping[str, Any],
    ) -> tuple[Path, Path]:
        """Write the monitor-ready CSV and a JSON manifest pointing at it."""

        json_path, csv_path = self.resolve_dataset_output_paths(
            json_path=json_path,
            csv_path=csv_path,
            output_prefix=output_prefix,
            class_count=class_count,
            sample_size=sample_size,
        )

        json_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        frame, dropped_count = filter_monitorable_rows(frame)
        frame.to_csv(csv_path, index=False)

        metadata_dict = dict(metadata)
        metadata_dict["rows_dropped_unmonitorable"] = dropped_count
        metadata_dict["monitorable_row_count"] = int(len(frame))

        manifest = {
            "csv_file": str(csv_path),
            "row_count": int(len(frame)),
            "columns": list(frame.columns),
            "openrouter": {
                "chat_model": self.chat_model,
                "embedding_model": self.embedding_model,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "top_logprobs": self.top_logprobs,
                "max_workers": self.max_workers,
                "chat_provider": self.chat_provider,
                "embedding_provider": self.embedding_provider,
                "reasoning": self.reasoning,
            },
            "metadata": metadata_dict,
        }
        json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return json_path, csv_path

    def warn_if_dataset_outputs_exist(
        self,
        *,
        json_path: Optional[Path] = None,
        csv_path: Optional[Path] = None,
        output_prefix: Optional[Path] = None,
        class_count: Optional[int] = None,
        sample_size: Optional[int] = None,
    ) -> tuple[Path, Path]:
        """Warn if the resolved CSV or JSON output paths already exist."""

        json_path, csv_path = self.resolve_dataset_output_paths(
            json_path=json_path,
            csv_path=csv_path,
            output_prefix=output_prefix,
            class_count=class_count,
            sample_size=sample_size,
        )
        existing = [path for path in (csv_path, json_path) if path.exists()]
        if existing:
            print(
                "Warning: output file(s) already exist and will be overwritten:\n"
                + "\n".join(f"  {path}" for path in existing),
                file=sys.stderr,
            )
        return json_path, csv_path

    def resolve_dataset_output_paths(
        self,
        *,
        json_path: Optional[Path] = None,
        csv_path: Optional[Path] = None,
        output_prefix: Optional[Path] = None,
        class_count: Optional[int] = None,
        sample_size: Optional[int] = None,
    ) -> tuple[Path, Path]:
        """Resolve deterministic CSV/JSON output paths for a text dataset run."""

        if json_path is not None and csv_path is None:
            return json_path, json_path.with_suffix(".csv")
        if csv_path is not None and json_path is None:
            return csv_path.with_suffix(".json"), csv_path
        if json_path is not None and csv_path is not None:
            return json_path, csv_path

        if output_prefix is None:
            raise ValueError("output_prefix is required when output paths are not explicit")
        if class_count is None:
            raise ValueError("class_count is required when output paths are not explicit")
        if sample_size is None:
            raise ValueError("sample_size is required when output paths are not explicit")

        stem = (
            f"{output_prefix.name}"
            f"judge-{model_name_slug(self.chat_model)}_"
            f"embed-{model_name_slug(self.embedding_model)}_"
            f"temp-{temperature_slug(self.temperature)}_"
            f"{class_count}class_"
            f"n{sample_size}"
        )
        csv_path = output_prefix.with_name(stem).with_suffix(".csv")
        json_path = csv_path.with_suffix(".json")
        return json_path, csv_path

    def _judge_one(
        self,
        prompt: str,
        *,
        label_tokens: Sequence[str],
        system_prompt: Optional[str],
    ) -> JudgeResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "stop": ["\n"],
        }
        if self.chat_provider is not None:
            payload["provider"] = self.chat_provider
        if self.reasoning is not None:
            payload["reasoning"] = self.reasoning

        response = self._post_json("/chat/completions", payload)
        if "choices" not in response or not response["choices"]:
            preview = json.dumps(response, ensure_ascii=False)[:1000]
            raise RuntimeError(f"OpenRouter response missing choices: {preview}")
        choice = response["choices"][0]
        answer = str(choice.get("message", {}).get("content") or "").strip()
        first_token_info = self._extract_first_token_info(choice)
        label_logprobs, label_sources, label_floor = self._extract_label_logprobs(
            choice,
            label_tokens=label_tokens,
        )
        label_probs = self._softmax_label_logprobs(label_logprobs)
        return JudgeResult(
            answer=answer,
            first_token=first_token_info["token"],
            first_token_logprob=first_token_info["logprob"],
            top_logprobs=first_token_info["top_logprobs"],
            label_logprobs=label_logprobs,
            label_logprob_sources=label_sources,
            label_logprob_floor=label_floor,
            label_probs=label_probs,
            finish_reason=choice.get("finish_reason"),
            model=response.get("model"),
            response_id=response.get("id"),
        )

    def _post_json(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        url = f"{OPENROUTER_BASE_URL}{path}"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_name,
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url

        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            request = Request(url, data=body, headers=headers, method="POST")
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                return json.loads(raw)
            except HTTPError as exc:
                last_error = exc
                if exc.code not in {408, 429, 500, 502, 503, 529}:
                    detail = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
            except URLError as exc:
                last_error = exc

            if attempt < self.max_retries:
                retry_after = retry_after_seconds(last_error)
                if retry_after is None:
                    retry_after = min(2.0**attempt, 30.0)
                time.sleep(retry_after)

        raise RuntimeError("OpenRouter request failed after retries") from last_error

    @staticmethod
    def _extract_first_token_info(choice: Mapping[str, Any]) -> dict[str, Any]:
        content_logprobs = (choice.get("logprobs") or {}).get("content") or []
        if not content_logprobs:
            return {"token": None, "logprob": None, "top_logprobs": ()}

        first_token = content_logprobs[0]
        top_logprobs = tuple(
            {
                "token": item.get("token"),
                "bytes": item.get("bytes"),
                "logprob": item.get("logprob"),
            }
            for item in (first_token.get("top_logprobs") or [])
        )
        return {
            "token": first_token.get("token"),
            "logprob": first_token.get("logprob"),
            "top_logprobs": top_logprobs,
        }

    @staticmethod
    def _extract_label_logprobs(
        choice: Mapping[str, Any],
        *,
        label_tokens: Sequence[str],
    ) -> tuple[dict[str, Optional[float]], dict[str, str], Optional[float]]:
        label_set = set(label_tokens)
        scores: dict[str, Optional[float]] = {label: None for label in label_tokens}
        sources: dict[str, str] = {label: "missing" for label in label_tokens}
        content_logprobs = (choice.get("logprobs") or {}).get("content") or []
        if not content_logprobs:
            return scores, sources, None

        first_token = content_logprobs[0]
        top_logprobs = first_token.get("top_logprobs") or []
        candidates = [first_token, *top_logprobs]
        candidate_logprobs = [
            float(item["logprob"]) for item in top_logprobs if item.get("logprob") is not None
        ]
        floor = min(candidate_logprobs) if candidate_logprobs else None
        for item in candidates:
            token = str(item.get("token", ""))
            normalized = token.strip()
            if normalized in label_set and scores[normalized] is None:
                scores[normalized] = float(item["logprob"])
                sources[normalized] = "exact"

        missing = [label for label, score in scores.items() if score is None]
        if floor is not None:
            for label in missing:
                scores[label] = floor
                sources[label] = "inferred_top_logprobs_floor"

        return scores, sources, floor

    @staticmethod
    def _softmax_label_logprobs(
        label_logprobs: Mapping[str, Optional[float]]
    ) -> dict[str, Optional[float]]:
        if any(value is None for value in label_logprobs.values()):
            return {label: None for label in label_logprobs}

        values = {label: float(value) for label, value in label_logprobs.items()}
        max_logprob = max(values.values())
        exps = {label: math.exp(value - max_logprob) for label, value in values.items()}
        denom = sum(exps.values())
        return {label: exps[label] / denom for label in values}

    @staticmethod
    def judge_result_to_columns(result: JudgeResult) -> dict[str, Any]:
        """Flatten a JudgeResult into CSV-friendly columns."""

        row: dict[str, Any] = {
            "judge_answer": result.answer,
            "first_token": result.first_token,
            "first_token_logprob": result.first_token_logprob,
            "top_logprobs_json": json.dumps(result.top_logprobs, ensure_ascii=False),
            "label_logprob_floor": result.label_logprob_floor,
            "judge_finish_reason": result.finish_reason,
            "judge_model_returned": result.model,
            "judge_response_id": result.response_id,
            "judge_error": result.error,
        }
        for label, value in result.label_logprobs.items():
            row[f"logprob_{label}"] = value
            row[f"logprob_{label}_source"] = result.label_logprob_sources[label]
        for label, value in result.label_probs.items():
            row[f"prob_{label}"] = value
        return row

    @staticmethod
    def judge_result_asdict(result: JudgeResult) -> dict[str, Any]:
        return asdict(result)


def load_env_file(explicit_path: Optional[Path] = None) -> None:
    """Load OPENROUTER_API_KEY from a simple .env file if present.

    Existing environment variables win. The parser intentionally supports only
    KEY=VALUE lines, which is enough for local experiment credentials.
    """

    for path in candidate_env_paths(explicit_path):
        if not path.exists():
            continue
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            os.environ.setdefault(key, value)


def filter_monitorable_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows that cannot be consumed by the quantitative monitor."""

    required = sorted(
        [col for col in frame.columns if col.startswith("prob_")],
        key=probability_column_sort_key,
    )
    if not required:
        return frame, 0

    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    mask = numeric.notna().all(axis=1)
    dropped = int((~mask).sum())
    if dropped == 0:
        return frame, 0
    return frame.loc[mask].copy(), dropped


def probability_column_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix("prob_")
    try:
        return (int(suffix), suffix)
    except ValueError:
        return (10**9, suffix)


def model_name_slug(model: str) -> str:
    """Return a filename-safe model slug without the OpenRouter provider prefix."""

    short = model.split("/", 1)[-1]
    slug = re.sub(r"[^A-Za-z0-9]+", "-", short).strip("-").lower()
    return slug or "model"


def temperature_slug(temperature: float) -> str:
    """Return a compact filename-safe temperature slug."""

    return "t" + ("%g" % float(temperature)).replace("-", "m").replace(".", "p")


def candidate_env_paths(explicit_path: Optional[Path] = None) -> list[Path]:
    if explicit_path is not None:
        return [explicit_path]

    module_dir = Path(__file__).resolve().parent
    candidates = [Path.cwd() / ".env", module_dir / ".env"]
    candidates.extend(parent / ".env" for parent in module_dir.parents[:3])

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def retry_after_seconds(error: Optional[BaseException]) -> Optional[float]:
    if not isinstance(error, HTTPError):
        return None
    value = error.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None
