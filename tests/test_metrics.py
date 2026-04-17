"""Tests for metrics parsing."""

from __future__ import annotations

from collections import deque

import pytest

from vllm_monitor.metrics import MetricsPoller, VllmMetrics, _parse_prometheus, sparkline

SAMPLE_PROMETHEUS = """
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="llama3"} 3.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="llama3"} 7.0
# HELP vllm:gpu_cache_usage_perc GPU KV cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="llama3"} 0.42
# HELP vllm:gpu_prefix_cache_hit_rate GPU prefix cache hit rate.
# TYPE vllm:gpu_prefix_cache_hit_rate gauge
vllm:gpu_prefix_cache_hit_rate{model_name="llama3"} 0.75
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="llama3"} 12000.0
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="llama3"} 45000.0
# HELP vllm:e2e_request_latency_seconds Histogram of e2e request latency in seconds.
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_sum{model_name="llama3"} 320.5
vllm:e2e_request_latency_seconds_count{model_name="llama3"} 200.0
"""


def test_parse_prometheus_basic():
    raw = _parse_prometheus(SAMPLE_PROMETHEUS)
    assert any("num_requests_running" in k for k in raw)
    assert any("gpu_cache_usage_perc" in k for k in raw)


def test_parse_into_metrics():
    poller = MetricsPoller(base_url="http://localhost:8000")
    m = VllmMetrics()
    poller._parse_into(m, SAMPLE_PROMETHEUS)

    assert m.num_requests_running == pytest.approx(3.0)
    assert m.num_requests_waiting == pytest.approx(7.0)
    assert m.gpu_cache_usage_perc == pytest.approx(42.0)
    assert m.gpu_prefix_cache_hit_rate == pytest.approx(75.0)
    assert m.prompt_tokens_total == pytest.approx(12000.0)
    assert m.generation_tokens_total == pytest.approx(45000.0)
    assert m.e2e_latency_mean_s == pytest.approx(320.5 / 200.0)


def test_rate_computation():
    poller = MetricsPoller(base_url="http://localhost:8000")

    prev = VllmMetrics(timestamp=0.0, server_reachable=True, generation_tokens_total=1000.0, prompt_tokens_total=500.0)
    poller._prev_metrics = prev
    poller._prev_time = 0.0

    current = VllmMetrics(timestamp=2.0, server_reachable=True, generation_tokens_total=1200.0, prompt_tokens_total=600.0)
    poller._compute_rates(current)

    assert current.generation_tokens_per_sec == pytest.approx(100.0)
    assert current.prompt_tokens_per_sec == pytest.approx(50.0)


def test_sparkline_basic():
    data: deque[float] = deque([0, 1, 2, 3, 4, 5], maxlen=60)
    spark = sparkline(data, width=6)
    assert len(spark) == 6
    assert spark[0] == " "  # 0 maps to empty
    assert spark[-1] == "█"  # max maps to full


def test_sparkline_empty():
    data: deque[float] = deque(maxlen=60)
    spark = sparkline(data, width=10)
    assert spark == " " * 10


def test_sparkline_all_same():
    data: deque[float] = deque([5.0] * 10, maxlen=60)
    spark = sparkline(data, width=10)
    assert len(spark) == 10
    # all same value → all map to max bar (since max_val == value)
    assert all(c == "█" for c in spark)
