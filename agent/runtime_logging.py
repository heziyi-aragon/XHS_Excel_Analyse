"""运行日志：记录步骤与错误类型，不记录请求、响应或认证信息。"""
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
import logging
import sys
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit
from uuid import uuid4
from zoneinfo import ZoneInfo


LOGGER = logging.getLogger("seedscout.runtime")
LOGGER.addHandler(logging.NullHandler())
LOGGER.propagate = False
CURRENT_NODE = ContextVar("runtime_node", default="standalone")


def error_chain(error: BaseException) -> str:
    """只保留异常类、系统错误号和HTTP状态，不序列化SDK错误正文。"""
    parts = []
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        detail = type(error).__name__
        for attribute in ("errno", "status_code", "verify_code"):
            value = getattr(error, attribute, None)
            if isinstance(value, int):
                detail += f" {attribute}={value}"
        parts.append(detail)
        error = error.__cause__ or error.__context__
    return " <- ".join(parts)


class HongKongFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created, ZoneInfo("Asia/Hong_Kong")).isoformat(timespec="seconds")


NODE_LABELS = {
    "load_inputs": "第1步 · 读取需求和达人数据",
    "analyze_requirements": "第2步 · 需求分析",
    "request_client_confirmation": "第2步 · 等待客户补充需求",
    "build_screening_policy": "第3步 · 生成筛选策略",
    "review_policy": "第3步 · 人工审核筛选策略",
    "prepare_next_step": "第3步 · 准备执行筛选策略",
    "preprocess_table": "第4步 · 数据清洗",
    "screen_talents": "第5步 · 筛选达人",
    "relax_screening": "第5步 · 放宽条件后重新筛选",
    "score_talents": "第6步 · 评分分档",
    "write_one_reason": "第7步 · 生成单个达人理由",
    "collect_reasons": "第7步 · 汇总达人理由",
    "audit_reasons": "第8步 · 审核达人理由",
    "prepare_reason_rewrites": "第8步 · 准备重写理由",
    "supplement_data": "第8步 · 补充数据",
    "finalize_audit": "第8步 · 汇总审核结果",
    "build_portfolio": "第9步 · 组合与定价",
    "review_portfolio": "第9步 · 报价复核",
    "write_advice": "第10步 · 写建议",
    "human_review": "第11步 · 人工审核与导出",
}


class TerminalProgressHandler(logging.Handler):
    """只输出步骤事件，不把详细日志或业务正文打印到终端。"""

    def __init__(self):
        super().__init__()
        self.reason_total = 0
        self.reason_done = 0
        self.reason_started = 0.0

    def emit(self, record):
        if record.msg == "reasons.batch total=%s":
            self.reason_total = record.args[0]
            self.reason_done = 0
            self.reason_started = perf_counter()
            if self.reason_total:
                print(f"[开始] 第7步 · 生成达人理由，共{self.reason_total}位", flush=True)
            return
        event = record.msg.split(" ", 1)[0]
        if event == "%s" and record.args:
            event = record.args[0]
        if event not in {"node.start", "node.end", "node.pause", "node.error", "node.business_error"}:
            return
        offset = 1 if record.msg.startswith("%s ") else 0
        name = record.args[offset]
        if name == "write_one_reason" and event in {"node.start", "node.end"}:
            if event == "node.end" and self.reason_total:
                self.reason_done += 1
                total = self.reason_total
                # 最多约四条进度；并行事件由logging.Handler的锁串行处理。
                interval = max(1, (total + 3) // 4)
                if self.reason_done == total:
                    elapsed = perf_counter() - self.reason_started
                    print(f"[结束] 第7步 · 达人理由处理完成 {total}/{total} · 耗时 {elapsed:.2f} 秒", flush=True)
                elif self.reason_done % interval == 0:
                    print(f"[进度] 第7步 · 已处理 {self.reason_done}/{total}", flush=True)
            return
        if name == "collect_reasons" and self.reason_total and event in {"node.start", "node.end"}:
            if event == "node.end":
                self.reason_total = 0
            return
        label = NODE_LABELS.get(name, name)
        if event == "node.start":
            message = f"[开始] {label}"
        elif event == "node.business_error":
            message = f"[需处理] {label} · 业务结果存在问题，请查看结果和日志"
        else:
            status = {"node.end": "结束", "node.pause": "等待人工输入", "node.error": "失败"}[event]
            message = f"[{status}] {label} · 耗时 {record.args[offset + 1]:.2f} 秒"
        print(message, file=sys.stdout, flush=True)


def setup_runtime_logging(directory: str | Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y%m%d_%H%M%S")
    path = directory / f"run_{stamp}_{uuid4().hex[:8]}.log"
    for handler in list(LOGGER.handlers):
        LOGGER.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(path, encoding="utf-8", mode="x")
    handler.setFormatter(HongKongFormatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.addHandler(TerminalProgressHandler())
    LOGGER.setLevel(logging.INFO)
    LOGGER.info("run.start")
    return path


def logged_node(name, function):
    @wraps(function)
    def run(state):
        token = CURRENT_NODE.set(name)
        started = perf_counter()
        LOGGER.info("node.start node=%s", name)
        try:
            result = function(state)
        except BaseException as exc:
            event = "node.pause" if type(exc).__name__ == "GraphInterrupt" else "node.error"
            LOGGER.log(logging.INFO if event == "node.pause" else logging.ERROR,
                       "%s node=%s elapsed=%.3fs error=%s", event, name,
                       perf_counter() - started, error_chain(exc))
            raise
        else:
            LOGGER.info("node.end node=%s elapsed=%.3fs status=%s", name,
                        perf_counter() - started,
                        result.get("workflow_status", "ok") if isinstance(result, dict) else "ok")
            if isinstance(result, dict) and result.get("portfolio_error"):
                LOGGER.warning("node.business_error node=%s type=%s", name,
                               result["portfolio_error"].get("type", "unknown"))
            return result
        finally:
            CURRENT_NODE.reset(token)
    return run


def add_logged_node(graph, name, function):
    graph.add_node(name, logged_node(name, function))


def log_llm_call(function):
    @wraps(function)
    def call(self, system_prompt, payload):
        started = perf_counter()
        request_id = uuid4().hex[:12]
        node = CURRENT_NODE.get()
        LOGGER.info("llm.start request=%s node=%s model=%s host=%s transport=%s trust_env=%s stream=%s max_retries=%s",
                    request_id, node, self.model, urlsplit(self.base_url).hostname or "default",
                    "offline" if self.transport is not None else "api", self.trust_env,
                    self.stream, self.max_retries if self.max_retries is not None else "sdk_default")
        try:
            result = function(self, system_prompt, payload)
        except Exception as exc:
            LOGGER.error("llm.error request=%s node=%s elapsed=%.3fs error=%s",
                         request_id, node, perf_counter() - started, error_chain(exc))
            raise
        LOGGER.info("llm.end request=%s node=%s elapsed=%.3fs", request_id, node, perf_counter() - started)
        return result
    return call
