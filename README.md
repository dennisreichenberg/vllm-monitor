# vllm-monitor

Real-time terminal UI dashboard for monitoring [vLLM](https://github.com/vllm-project/vllm) server metrics. No Grafana required.

```
┌─────────────────────────────────── vllm-monitor ───────────────────────────────────┐
│ ● ONLINE  http://localhost:8000  interval=2s                                        │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────────────┤
│ Model Info      │ Running Req      │ Queued Req       │ Avg E2E Latency             │
│ llama3          │       3          │       7          │       1600ms                │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┬───────────┤
│ Prompt Tok/s    │ Gen Tok/s        │ GPU KV Cache     │ Prefix Hit       │ GPU Mem   │
│    450.2        │    312.7         │   42.0%          │   75.0%          │  68.4%    │
├─────────────────┴─────────────────┴──────────────────┴─────────────────┴───────────┤
│ Active Reqs  ▁▂▂▃▃▄▄▅▅▅▆▆▇▇█ current=3                                             │
│ Gen Tok/s    ▁▁▂▃▃▄▅▅▆▆▇▇███ current=312.7 tok/s                                   │
│ GPU Cache %  ▂▂▃▃▄▄▄▅▅▅▅▆▆▆▆ current=42.0%                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
  q Quit  r Refresh now
```

## Features

- **Real-time metrics**: requests/sec, active + queued requests, token throughput (prompt & generated)
- **Cache stats**: GPU KV cache utilization, prefix cache hit rate
- **GPU memory**: used vs total (when exposed by vLLM)
- **Request history sparklines**: rolling 60-sample charts for active requests, token throughput, and cache usage
- **Alert colors**: yellow at 80%, red at 90% for GPU cache and memory
- **Model info panel**: loaded model name from `/v1/models`
- **Configurable poll interval** (default 2s)

## Installation

```bash
pip install vllm-monitor
```

## Usage

```bash
# Monitor local vLLM server (default: http://localhost:8000, 2s interval)
vllm-monitor

# Custom server URL
vllm-monitor --url http://my-vllm-server:8000

# Faster refresh
vllm-monitor --interval 1

# With API key
vllm-monitor --url http://my-server:8000 --api-key mytoken
# or via env var:
VLLM_API_KEY=mytoken vllm-monitor
```

### Key Bindings

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Refresh immediately |

## Metrics Displayed

| Metric | Source | Description |
|--------|--------|-------------|
| Running Requests | `vllm:num_requests_running` | Requests actively being processed |
| Queued Requests | `vllm:num_requests_waiting` | Requests waiting for GPU capacity |
| Avg E2E Latency | `vllm:e2e_request_latency_seconds` | Mean end-to-end request latency |
| Prompt Tokens/s | `vllm:prompt_tokens_total` (rate) | Prompt token processing throughput |
| Gen Tokens/s | `vllm:generation_tokens_total` (rate) | Token generation throughput |
| GPU KV Cache | `vllm:gpu_cache_usage_perc` | KV cache block utilization |
| Prefix Cache Hit | `vllm:gpu_prefix_cache_hit_rate` | Prefix (radix) cache hit rate |
| GPU Memory | `vllm:gpu_memory_*_bytes` | GPU VRAM utilization |

## Requirements

- Python 3.10+
- vLLM server with `/metrics` (Prometheus) and `/v1/models` endpoints enabled (default)

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
