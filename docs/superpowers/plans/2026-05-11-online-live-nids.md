# 在线运行型 NIDS 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前离线 `UNSW-NB15` 实验项目改造成可运行的在线 NIDS：支持抓包、实时流聚合、模型打分与实时告警。

**Architecture:** 不直接把当前离线模型强行套到在线抓包场景。新增一条在线运行链路：使用 `scapy` 抓包，在内存中做双向流聚合，只提取那些确实能从实时流量中推导出来的特征子集，并基于该特征子集重训一个专用二分类模型，供实时推理使用。现有离线实验流水线保持不变。

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn, xgboost, scapy, joblib, pytest

---

## 范围决策

本计划默认采用“**在线特征子集 + 重训练在线模型**”这条路线，因为它是当前最小、可验证且技术上成立的方案。

不采用的方案：

- 直接复用现有离线模型：不可取，因为当前很多 `UNSW-NB15` 特征无法从实时原始包中直接得到。
- 一开始就先做完整的 Zeek/Argus 风格流量引擎：准确性会更高，但对当前仓库来说改动过大，不适合第一阶段落地。

## 文件结构

**新增**

- `src/ids_ml/live/__init__.py`：在线运行子包标记
- `src/ids_ml/live/config.py`：在线运行常量、特征模式、告警阈值、流超时配置
- `src/ids_ml/live/features.py`：在线特征子集定义与特征行构建
- `src/ids_ml/live/flows.py`：从数据包聚合为流，维护双向流状态
- `src/ids_ml/live/alerts.py`：告警数据结构与输出 sink
- `src/ids_ml/live/runtime.py`：模型加载、实时打分和告警逻辑
- `src/ids_ml/pipeline_online_train.py`：基于在线可推导特征重训二分类模型
- `src/ids_ml/pipeline_live.py`：实时抓包与在线推理 CLI 入口
- `tests/test_online_train.py`：在线训练链路测试
- `tests/test_live_features.py`：在线特征与流聚合测试
- `tests/test_live_runtime.py`：实时打分与告警测试

**修改**

- `requirements.txt`：增加抓包依赖
- `README.md`：补充在线 NIDS 运行说明和权限要求

## 在线特征子集

第一版在线可运行版本只使用能从包头和简单流统计中推导出的特征：

- `proto`
- `service`
- `state`
- `dur`
- `spkts`
- `dpkts`
- `sbytes`
- `dbytes`
- `rate`
- `sttl`
- `dttl`
- `smean`
- `dmean`
- `sinpkt`
- `dinpkt`
- `ct_srv_dst`
- `ct_dst_src_ltm`
- `is_sm_ips_ports`

约定：

- `service` 优先根据常见目的端口推断，推断不了时默认 `"unknown"`
- `state` 采用最小可用映射：
  - 仅看到 TCP SYN：`SYN`
  - 双向看到 SYN/SYN+ACK：`CON`
  - 看到 TCP FIN 或 RST：`FIN`
  - UDP 或其他协议且已双向通信：`INT`
  - 其他情况：`UNK`

## 测试策略

- 先用合成的 packet-like 对象验证流聚合逻辑，再谈真实抓包。
- 单测覆盖特征构建、`service/state` 推断与列顺序稳定性。
- 单测覆盖运行期告警逻辑，使用固定输出概率的 fake model。
- 暂时不做强依赖真实网卡的 CLI 自动化测试，因为实时抓包需要宿主权限和接口环境。

### Task 1: 建立在线训练特征模式

**Files:**
- Create: `src/ids_ml/live/__init__.py`
- Create: `src/ids_ml/live/config.py`
- Create: `src/ids_ml/live/features.py`
- Test: `tests/test_live_features.py`

- [ ] **Step 1: 先写失败测试，固定在线特征集合与列顺序**

```python
import pandas as pd

from src.ids_ml.live.features import ONLINE_BINARY_FEATURES, build_online_feature_frame


def test_online_feature_list_is_stable():
    assert ONLINE_BINARY_FEATURES == [
        "proto",
        "service",
        "state",
        "dur",
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
        "rate",
        "sttl",
        "dttl",
        "smean",
        "dmean",
        "sinpkt",
        "dinpkt",
        "ct_srv_dst",
        "ct_dst_src_ltm",
        "is_sm_ips_ports",
    ]


def test_build_online_feature_frame_keeps_declared_column_order():
    frame = pd.DataFrame(
        [
            {
                "proto": "tcp",
                "service": "http",
                "state": "CON",
                "dur": 1.0,
                "spkts": 3,
                "dpkts": 2,
                "sbytes": 120,
                "dbytes": 80,
                "rate": 5.0,
                "sttl": 64,
                "dttl": 63,
                "smean": 40.0,
                "dmean": 40.0,
                "sinpkt": 0.2,
                "dinpkt": 0.3,
                "ct_srv_dst": 1,
                "ct_dst_src_ltm": 1,
                "is_sm_ips_ports": 0,
                "label": 1,
            }
        ]
    )

    online = build_online_feature_frame(frame)

    assert online.columns.tolist() == ONLINE_BINARY_FEATURES
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run: `pytest tests/test_live_features.py -v`
Expected: FAIL with import error for `src.ids_ml.live.features`

- [ ] **Step 3: 写最小实现，建立在线特征模式**

```python
from dataclasses import dataclass

import pandas as pd


ONLINE_BINARY_FEATURES = [
    "proto",
    "service",
    "state",
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "smean",
    "dmean",
    "sinpkt",
    "dinpkt",
    "ct_srv_dst",
    "ct_dst_src_ltm",
    "is_sm_ips_ports",
]


def build_online_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["service"] = prepared["service"].fillna("unknown").astype(str)
    prepared["state"] = prepared["state"].fillna("UNK").astype(str)
    return prepared.reindex(columns=ONLINE_BINARY_FEATURES)
```

- [ ] **Step 4: 再跑测试，确认通过**

Run: `pytest tests/test_live_features.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/ids_ml/live/__init__.py src/ids_ml/live/config.py src/ids_ml/live/features.py tests/test_live_features.py
git commit -m "feat: add online feature schema"
```

### Task 2: 增加在线可兼容训练流水线

**Files:**
- Create: `src/ids_ml/pipeline_online_train.py`
- Modify: `src/ids_ml/train.py`
- Test: `tests/test_online_train.py`

- [ ] **Step 1: 先写失败测试，约束在线训练数据准备逻辑**

```python
import pandas as pd

from src.ids_ml.pipeline_online_train import prepare_online_binary_training_frame


def test_prepare_online_binary_training_frame_keeps_only_online_features_and_label():
    frame = pd.DataFrame(
        [
            {
                "proto": "tcp",
                "service": "http",
                "state": "CON",
                "dur": 1.0,
                "spkts": 3,
                "dpkts": 2,
                "sbytes": 120,
                "dbytes": 80,
                "rate": 5.0,
                "sttl": 64,
                "dttl": 63,
                "smean": 40.0,
                "dmean": 40.0,
                "sinpkt": 0.2,
                "dinpkt": 0.3,
                "ct_srv_dst": 1,
                "ct_dst_src_ltm": 1,
                "is_sm_ips_ports": 0,
                "label": 1,
                "attack_cat": "Generic",
                "unused": 999,
            }
        ]
    )

    prepared = prepare_online_binary_training_frame(frame)

    assert prepared.columns.tolist() == [
        "proto",
        "service",
        "state",
        "dur",
        "spkts",
        "dpkts",
        "sbytes",
        "dbytes",
        "rate",
        "sttl",
        "dttl",
        "smean",
        "dmean",
        "sinpkt",
        "dinpkt",
        "ct_srv_dst",
        "ct_dst_src_ltm",
        "is_sm_ips_ports",
        "label",
    ]
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run: `pytest tests/test_online_train.py::test_prepare_online_binary_training_frame_keeps_only_online_features_and_label -v`
Expected: FAIL with import error for `src.ids_ml.pipeline_online_train`

- [ ] **Step 3: 写最小在线训练流水线**

```python
import pandas as pd

from .config import BINARY_TARGET, MODELS_DIR
from .data import load_unsw_nb15
from .evaluate import compute_binary_metrics
from .live.features import ONLINE_BINARY_FEATURES, build_online_feature_frame
from .preprocess import split_binary_features_target
from .train import fit_model, get_binary_models, save_model


def prepare_online_binary_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    subset = frame[ONLINE_BINARY_FEATURES + [BINARY_TARGET]].copy()
    subset.loc[:, ONLINE_BINARY_FEATURES] = build_online_feature_frame(subset).values
    return subset
```

- [ ] **Step 4: 扩展为可训练并持久化在线专用模型**

```python
def main():
    train_df, test_df = load_unsw_nb15()
    train_df = prepare_online_binary_training_frame(train_df)
    test_df = prepare_online_binary_training_frame(test_df)
    x_train, y_train = split_binary_features_target(train_df)
    x_test, y_test = split_binary_features_target(test_df)

    estimator = get_binary_models()["xgboost"]
    pipeline = fit_model(build_preprocessor(x_train), estimator, x_train, y_train)
    y_pred = pipeline.predict(x_test)
    y_score = pipeline.predict_proba(x_test)[:, 1]
    metrics = compute_binary_metrics(y_test, y_pred, y_score)

    save_model(pipeline, MODELS_DIR / "online_binary.joblib")
    pd.DataFrame([{"model": "online_binary", **metrics}]).to_csv(
        "results/online_binary_metrics.csv",
        index=False,
    )
```

- [ ] **Step 5: 运行测试，确认准备逻辑通过**

Run: `pytest tests/test_online_train.py -v`
Expected: PASS

- [ ] **Step 6: 运行训练流水线**

Run: `python -m src.ids_ml.pipeline_online_train`
Expected: `models/online_binary.joblib` and `results/online_binary_metrics.csv`

- [ ] **Step 7: 提交**

```bash
git add src/ids_ml/pipeline_online_train.py tests/test_online_train.py results/online_binary_metrics.csv models/online_binary.joblib
git commit -m "feat: add online-compatible binary training pipeline"
```

### Task 3: 实现实时流聚合

**Files:**
- Create: `src/ids_ml/live/flows.py`
- Test: `tests/test_live_features.py`

- [ ] **Step 1: 先写失败测试，约束双向流统计**

```python
from src.ids_ml.live.flows import FlowTable, PacketEvent


def test_flow_table_tracks_forward_and_reverse_counters():
    table = FlowTable(flow_timeout_seconds=30)

    table.consume(
        PacketEvent(
            timestamp=1.0,
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            src_port=12345,
            dst_port=80,
            protocol="tcp",
            size=60,
            ttl=64,
            tcp_flags={"SYN"},
        )
    )
    table.consume(
        PacketEvent(
            timestamp=1.2,
            src_ip="10.0.0.2",
            dst_ip="10.0.0.1",
            src_port=80,
            dst_port=12345,
            protocol="tcp",
            size=52,
            ttl=63,
            tcp_flags={"SYN", "ACK"},
        )
    )

    flow = next(iter(table.active_flows().values()))

    assert flow.spkts == 1
    assert flow.dpkts == 1
    assert flow.service == "http"
    assert flow.state == "CON"
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run: `pytest tests/test_live_features.py::test_flow_table_tracks_forward_and_reverse_counters -v`
Expected: FAIL with import error for `src.ids_ml.live.flows`

- [ ] **Step 3: 实现 packet event 和 flow table**

```python
from dataclasses import dataclass, field


@dataclass
class PacketEvent:
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    size: int
    ttl: int
    tcp_flags: set[str]


@dataclass
class FlowRecord:
    origin_src_ip: str
    origin_dst_ip: str
    origin_src_port: int
    origin_dst_port: int
    protocol: str
    started_at: float
    last_seen_at: float
    spkts: int = 0
    dpkts: int = 0
    sbytes: int = 0
    dbytes: int = 0
    sttl_values: list[int] = field(default_factory=list)
    dttl_values: list[int] = field(default_factory=list)
    service: str = "unknown"
    state: str = "UNK"
```

- [ ] **Step 4: 增加流转特征行和超时回收逻辑**

```python
def expire(self, now: float) -> list[FlowRecord]:
    expired = []
    for key, record in list(self._flows.items()):
        if now - record.last_seen_at >= self.flow_timeout_seconds:
            expired.append(self._flows.pop(key))
    return expired
```

- [ ] **Step 5: 运行测试**

Run: `pytest tests/test_live_features.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/ids_ml/live/flows.py tests/test_live_features.py
git commit -m "feat: add live flow aggregation"
```

### Task 4: 实现实时打分与告警

**Files:**
- Create: `src/ids_ml/live/alerts.py`
- Create: `src/ids_ml/live/runtime.py`
- Test: `tests/test_live_runtime.py`

- [ ] **Step 1: 先写失败测试，约束告警生成逻辑**

```python
from src.ids_ml.live.runtime import score_completed_flow


class FakeModel:
    def predict_proba(self, frame):
        return [[0.02, 0.98]]


def test_score_completed_flow_emits_alert_when_threshold_is_crossed():
    feature_row = {
        "proto": "tcp",
        "service": "http",
        "state": "CON",
        "dur": 1.0,
        "spkts": 10,
        "dpkts": 8,
        "sbytes": 800,
        "dbytes": 600,
        "rate": 18.0,
        "sttl": 64,
        "dttl": 63,
        "smean": 80.0,
        "dmean": 75.0,
        "sinpkt": 0.1,
        "dinpkt": 0.12,
        "ct_srv_dst": 4,
        "ct_dst_src_ltm": 3,
        "is_sm_ips_ports": 0,
    }

    alert = score_completed_flow(FakeModel(), feature_row, alert_threshold=0.9)

    assert alert is not None
    assert alert.score == 0.98
```

- [ ] **Step 2: 运行测试，确认当前失败**

Run: `pytest tests/test_live_runtime.py -v`
Expected: FAIL with import error for `src.ids_ml.live.runtime`

- [ ] **Step 3: 实现告警数据结构和打分函数**

```python
from dataclasses import dataclass

import pandas as pd

from .features import ONLINE_BINARY_FEATURES


@dataclass
class AlertEvent:
    score: float
    threshold: float
    feature_row: dict


def score_completed_flow(model, feature_row: dict, alert_threshold: float):
    frame = pd.DataFrame([feature_row], columns=ONLINE_BINARY_FEATURES)
    score = float(model.predict_proba(frame)[0][1])
    if score < alert_threshold:
        return None
    return AlertEvent(score=score, threshold=alert_threshold, feature_row=feature_row)
```

- [ ] **Step 4: 实现 console 和 JSONL 告警 sink**

```python
import json
from dataclasses import asdict
from pathlib import Path


def emit_console_alert(alert):
    print(
        f"[ALERT] score={alert.score:.4f} threshold={alert.threshold:.2f} "
        f"service={alert.feature_row['service']} state={alert.feature_row['state']}"
    )


def append_jsonl_alert(alert, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(alert), ensure_ascii=False) + "\n")
```

- [ ] **Step 5: 运行测试**

Run: `pytest tests/test_live_runtime.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add src/ids_ml/live/alerts.py src/ids_ml/live/runtime.py tests/test_live_runtime.py
git commit -m "feat: add live runtime scoring and alert sinks"
```

### Task 5: 增加实时抓包 CLI

**Files:**
- Create: `src/ids_ml/pipeline_live.py`
- Modify: `requirements.txt`
- Modify: `README.md`

- [ ] **Step 1: 增加抓包依赖**

```txt
scapy>=2.6
```

- [ ] **Step 2: 实现 CLI 参数与实时循环入口**

```python
import argparse
import time

import joblib
from scapy.all import AsyncSniffer, IP, TCP, UDP

from .live.alerts import append_jsonl_alert, emit_console_alert
from .live.flows import FlowTable, PacketEvent
from .live.runtime import score_completed_flow


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", required=True)
    parser.add_argument("--model-path", default="models/online_binary.joblib")
    parser.add_argument("--alert-threshold", type=float, default=0.90)
    parser.add_argument("--flow-timeout", type=int, default=15)
    parser.add_argument("--alerts-path", default="results/live_alerts.jsonl")
    return parser.parse_args()
```

- [ ] **Step 3: 增加 packet 标准化回调**

```python
def packet_to_event(packet, timestamp):
    if IP not in packet:
        return None
    transport = packet[TCP] if TCP in packet else packet[UDP] if UDP in packet else None
    if transport is None:
        return None
    return PacketEvent(
        timestamp=timestamp,
        src_ip=packet[IP].src,
        dst_ip=packet[IP].dst,
        src_port=int(getattr(transport, "sport", 0)),
        dst_port=int(getattr(transport, "dport", 0)),
        protocol="tcp" if TCP in packet else "udp",
        size=len(packet),
        ttl=int(packet[IP].ttl),
        tcp_flags=set(str(packet[TCP].flags)) if TCP in packet else set(),
    )
```

- [ ] **Step 4: 实现在线抓包主循环**

```python
def main():
    args = parse_args()
    model = joblib.load(args.model_path)
    table = FlowTable(flow_timeout_seconds=args.flow_timeout)

    def handle_packet(packet):
        event = packet_to_event(packet, time.time())
        if event is None:
            return
        table.consume(event)
        for flow in table.expire(event.timestamp):
            feature_row = flow.to_feature_row(table.snapshot_service_count(flow.origin_dst_ip, flow.service))
            alert = score_completed_flow(model, feature_row, args.alert_threshold)
            if alert is None:
                continue
            emit_console_alert(alert)
            append_jsonl_alert(alert, args.alerts_path)

    sniffer = AsyncSniffer(iface=args.iface, prn=handle_packet, store=False)
    sniffer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sniffer.stop()
```

- [ ] **Step 5: 在 `README.md` 中补充运行顺序**

```md
## Online Live NIDS

1. Train the online-compatible model: `python -m src.ids_ml.pipeline_online_train`
2. Run with administrator/Npcap support: `python -m src.ids_ml.pipeline_live --iface "<interface-name>"`
3. Watch alerts in console and `results/live_alerts.jsonl`

Notes:
- This path uses a reduced online feature subset, not the full offline feature space.
- Windows live capture requires packet capture support such as Npcap and an elevated shell.
```

- [ ] **Step 6: 运行定向测试**

Run: `pytest tests/test_live_features.py tests/test_online_train.py tests/test_live_runtime.py -q`
Expected: PASS

- [ ] **Step 7: 手工烟测**

Run: `python -m src.ids_ml.pipeline_live --iface "<your-interface-name>" --alert-threshold 0.9`
Expected: process starts, waits for packets, writes alerts when high-score flows expire

- [ ] **Step 8: 提交**

```bash
git add requirements.txt README.md src/ids_ml/pipeline_live.py
git commit -m "feat: add live packet capture and alert pipeline"
```

## Self-Review

- 需求覆盖：已覆盖在线抓包、实时告警、在线兼容模型、流聚合和运行说明。
- 占位符扫描：没有保留 `TODO`、`稍后实现` 或“自行处理”这种无执行细节的描述。
- 类型一致性：全程统一使用 `ONLINE_BINARY_FEATURES`、`PacketEvent`、`FlowTable`、`AlertEvent`、`score_completed_flow`、`pipeline_online_train` 和 `pipeline_live`。

## Notes for Execution

- 开始实现前先切到非 `main` 分支。
- 没有真实网卡烟测证据时，不能声称实时抓包链路已经可用。
- 如果 `scapy` 因为缺失驱动无法抓包，应先安装或验证 `Npcap`，不要绕过这个前提硬改代码。
