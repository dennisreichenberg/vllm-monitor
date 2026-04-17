"""Metrics polling and parsing for vLLM server."""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import httpx

# Maximum history samples kept for sparkline
HISTORY_SIZE = 60


@dataclass
class ModelInfo:
    model_id: str = "unknown"
    max_model_len: Optional[int] = None
    tensor_parallel_size: Optional[int] = None


@dataclass
class VllmMetrics:
    # Server state
    timestamp: float = 0.0
    server_reachable: bool = False

    # Request metrics
    num_requests_running: float = 0.0
    num_requests_waiting: float = 0.0
    num_requests_swapped: float = 0.0
    request_success_total: float = 0.0

    # Token throughput (tokens/sec, computed as delta)
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0
    prompt_tokens_per_sec: float = 0.0
    generation_tokens_per_sec: float = 0.0

    # Cache
    gpu_cache_usage_perc: float = 0.0
    cpu_cache_usage_perc: float = 0.0
    gpu_prefix_cache_hit_rate: float = 0.0

    # GPU memory (filled from /metrics if available)
    gpu_memory_used_bytes: float = 0.0
    gpu_memory_total_bytes: float = 0.0

    # Latency
    e2e_latency_mean_s: float = 0.0

    # Model info
    model_info: ModelInfo = field(default_factory=ModelInfo)


@dataclass
class MetricsHistory:
    requests_running: deque[float] = field(default_factory=lambda: deque([0.0] * HISTORY_SIZE, maxlen=HISTORY_SIZE))
    generation_tps: deque[float] = field(default_factory=lambda: deque([0.0] * HISTORY_SIZE, maxlen=HISTORY_SIZE))
    gpu_cache: deque[float] = field(default_factory=lambda: deque([0.0] * HISTORY_SIZE, maxlen=HISTORY_SIZE))


def _parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus text format into a flat metric name → value dict."""
    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Match metric_name{labels} value or metric_name value
        m = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?|NaN|[+-]?Inf)\s*$', line)
        if m:
            name = m.group(1)
            try:
                result[name] = float(m.group(2))
            except ValueError:
                pass
    return result


def _get_gauge(raw: dict[str, float], *keys: str) -> float:
    for k in keys:
        if k in raw:
            return raw[k]
        # Also try without labels
        for rk in raw:
            if rk.startswith(k + "{") or rk == k:
                return raw[rk]
    return 0.0


class MetricsPoller:
    def __init__(self, base_url: str, api_key: Optional[str] = None, interval: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.interval = interval
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(headers=headers, timeout=5.0)
        self._prev_metrics: Optional[VllmMetrics] = None
        self._prev_time: float = 0.0
        self.history = MetricsHistory()

    async def close(self) -> None:
        await self._client.aclose()

    async def poll(self) -> VllmMetrics:
        m = VllmMetrics(timestamp=time.time())
        try:
            prom_text = await self._fetch_prometheus()
            model_info = await self._fetch_model_info()
            m.server_reachable = True
            m.model_info = model_info
            self._parse_into(m, prom_text)
            self._compute_rates(m)
        except Exception:
            m.server_reachable = False

        self._update_history(m)
        self._prev_metrics = m
        self._prev_time = m.timestamp
        return m

    async def _fetch_prometheus(self) -> str:
        resp = await self._client.get(f"{self.base_url}/metrics")
        resp.raise_for_status()
        return resp.text

    async def _fetch_model_info(self) -> ModelInfo:
        try:
            resp = await self._client.get(f"{self.base_url}/v1/models")
            resp.raise_for_status()
            data = resp.json()
            models = data.get("data", [])
            if models:
                first = models[0]
                info = ModelInfo(model_id=first.get("id", "unknown"))
                perms = first.get("permission", [{}])
                if perms:
                    pass  # vLLM doesn't always expose context_length here
                return info
        except Exception:
            pass
        return ModelInfo()

    def _parse_into(self, m: VllmMetrics, text: str) -> None:
        raw = _parse_prometheus(text)

        m.num_requests_running = _get_gauge(raw, "vllm:num_requests_running")
        m.num_requests_waiting = _get_gauge(raw, "vllm:num_requests_waiting")
        m.num_requests_swapped = _get_gauge(raw, "vllm:num_requests_swapped")

        # Sum across all model labels for token totals
        prompt_total = 0.0
        gen_total = 0.0
        success_total = 0.0
        for k, v in raw.items():
            if "prompt_tokens_total" in k:
                prompt_total += v
            if "generation_tokens_total" in k:
                gen_total += v
            if "request_success_total" in k:
                success_total += v
        m.prompt_tokens_total = prompt_total
        m.generation_tokens_total = gen_total
        m.request_success_total = success_total

        m.gpu_cache_usage_perc = _get_gauge(raw, "vllm:gpu_cache_usage_perc") * 100
        m.cpu_cache_usage_perc = _get_gauge(raw, "vllm:cpu_cache_usage_perc") * 100
        m.gpu_prefix_cache_hit_rate = _get_gauge(raw, "vllm:gpu_prefix_cache_hit_rate") * 100

        # e2e latency bucket/sum — use _sum/_count if available
        latency_sum = 0.0
        latency_count = 0.0
        for k, v in raw.items():
            if "e2e_request_latency_seconds_sum" in k:
                latency_sum += v
            if "e2e_request_latency_seconds_count" in k:
                latency_count += v
        if latency_count > 0:
            m.e2e_latency_mean_s = latency_sum / latency_count

        # GPU memory
        for k, v in raw.items():
            if "gpu_memory_used_bytes" in k:
                m.gpu_memory_used_bytes = v
            if "gpu_memory_total_bytes" in k:
                m.gpu_memory_total_bytes = v

    def _compute_rates(self, current: VllmMetrics) -> None:
        if self._prev_metrics is None or not self._prev_metrics.server_reachable:
            return
        dt = current.timestamp - self._prev_metrics.timestamp
        if dt <= 0:
            return
        current.prompt_tokens_per_sec = max(0.0, (current.prompt_tokens_total - self._prev_metrics.prompt_tokens_total) / dt)
        current.generation_tokens_per_sec = max(0.0, (current.generation_tokens_total - self._prev_metrics.generation_tokens_total) / dt)

    def _update_history(self, m: VllmMetrics) -> None:
        self.history.requests_running.append(m.num_requests_running)
        self.history.generation_tps.append(m.generation_tokens_per_sec)
        self.history.gpu_cache.append(m.gpu_cache_usage_perc)


def sparkline(values: deque[float], width: int = 20) -> str:
    """Render a unicode block sparkline from a deque of floats."""
    bars = " ▁▂▃▄▅▆▇█"
    samples = list(values)[-width:]
    if not samples:
        return " " * width
    max_val = max(samples) or 1.0
    result = []
    for v in samples:
        idx = min(int(v / max_val * (len(bars) - 1)), len(bars) - 1)
        result.append(bars[idx])
    return "".join(result)
