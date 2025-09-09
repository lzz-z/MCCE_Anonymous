#!/usr/bin/env python3
"""
AlphaFold3 API 远程并行测试脚本（_4gpu 版）

支持一次传入最多4个分子（序列B），并发提交到服务端 /predict 接口，
分别监控各自的任务状态，最终汇总并保存每个任务的结果。

说明：该脚本仅做API层面的并行测试，不依赖服务端必须是4卡；
若服务端具备4卡并行能力，则可并行充分利用。
"""

import argparse
import concurrent.futures
import json
import sys
import time
from typing import List, Optional, Tuple

import requests


# 固定配置（如需修改，请直接编辑下方常量）
API_HOST = "192.168.8.169"
API_PORT = 8000
REQUEST_TIMEOUT = 30
POLL_INTERVAL = 15
MAX_WAIT_TIME = 1800
MAX_CLIENT_CONCURRENCY = 52


class MultiRemoteAPITester:
    def __init__(self, api_host: str, api_port: int = 8000, timeout: int = 30, poll_interval: int = 15):
        self.base_url = f"http://{api_host}:{api_port}"
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'AlphaFold3-Remote-Client/1.0-4gpu'
        })

        print("🌐 远程API并行测试客户端 (_4gpu)")
        print(f"目标服务器: {self.base_url}")
        print(f"请求超时: {self.timeout}秒, 轮询间隔: {self.poll_interval}秒")
        print("=" * 50)

    # 基础健康检查
    def test_connectivity(self) -> bool:
        print("=== 网络连通性测试 ===")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            print("✅ HTTP连接成功")
            print(f"响应状态码: {response.status_code}")
            print(f"响应时间: {response.elapsed.total_seconds():.2f}秒")
            return True
        except requests.exceptions.ConnectTimeout:
            print("❌ 连接超时 - 检查网络连接或防火墙设置")
            return False
        except requests.exceptions.ConnectionError as e:
            print(f"❌ 连接错误: {e}")
            print("可能的原因:\n  1. API服务器未启动\n  2. 防火墙阻止了连接\n  3. IP地址或端口错误")
            return False
        except Exception as e:
            print(f"❌ 其他错误: {e}")
            return False

    def test_health(self) -> bool:
        print("\n=== API健康检查 ===")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            health = response.json()
            print(f"健康状态: {health}")
            if health.get("status") == "healthy":
                print("✅ API服务健康")
                return True
            print("❌ API服务不健康")
            return False
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP错误: {e}")
            return False
        except requests.exceptions.Timeout:
            print("❌ 请求超时")
            return False
        except Exception as e:
            print(f"❌ 健康检查失败: {e}")
            return False

    # 单任务流程
    def submit_prediction(self, sequence: str, job_name: Optional[str] = None) -> Optional[str]:
        payload = {"sequence": sequence}
        if job_name:
            payload["job_name"] = job_name
        try:
            r = self.session.post(f"{self.base_url}/predict", json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("job_id")
        except requests.exceptions.HTTPError as e:
            print(f"❌ 提交任务HTTP错误: {e}")
            if hasattr(e.response, 'text'):
                print(f"错误详情: {e.response.text}")
            return None
        except Exception as e:
            print(f"❌ 提交任务失败: {e}")
            return None

    def monitor_job(self, job_id: str, max_wait_time: int = 1800) -> Optional[dict]:
        start = time.time()
        last_status = None
        check_count = 0
        while time.time() - start < max_wait_time:
            try:
                r = self.session.get(f"{self.base_url}/status/{job_id}", timeout=self.timeout)
                r.raise_for_status()
                info = r.json()
                status = info.get("status")
                msg = info.get("message", "")
                check_count += 1
                if status != last_status or check_count % max(1, int(90 / max(1, self.poll_interval))) == 1:
                    elapsed = time.time() - start
                    print(f"[{job_id}] 状态: {status} - {msg} (t={elapsed:.0f}s)")
                    last_status = status
                if status == "completed" or status == "failed":
                    return info
                time.sleep(self.poll_interval)
            except requests.exceptions.Timeout:
                print(f"[{job_id}] ⚠️ 查询超时，重试...")
                time.sleep(5)
            except Exception as e:
                print(f"[{job_id}] ⚠️ 查询异常: {e}")
                time.sleep(self.poll_interval)
        print(f"[{job_id}] ❌ 监控超时")
        return None

    def analyze_result(self, result_info: Optional[dict], tag: str) -> bool:
        if not result_info or result_info.get("status") != "completed":
            print(f"[{tag}] ❌ 无有效结果可分析")
            return False
        result = result_info.get("result", {})
        conf = result.get("summary_confidences", {})
        if not conf:
            print(f"[{tag}] ❌ 未找到置信度数据")
            return False
        iptm = conf.get('iptm', 'N/A')
        ptm = conf.get('ptm', 'N/A')
        ranking = conf.get('ranking_score', 'N/A')
        print(f"[{tag}] 🎯 置信度: iptm={iptm}, ptm={ptm}, ranking_score={ranking}")
        # 保存
        save_path = f"remote_result_{tag}.json"
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[{tag}] 💾 结果保存: {save_path}")
        except Exception as e:
            print(f"[{tag}] ⚠️ 保存结果失败: {e}")
        return True

    # 并行整体流程
    def run_parallel_test(
        self,
        sequences: List[str],
        names: Optional[List[str]] = None,
        max_wait_time: int = 1800,
    ) -> bool:
        print("🚀 开始远程AlphaFold3 API并行测试 (_4gpu)")
        print("=" * 60)

        if not self.test_connectivity():
            return False, {}
        if not self.test_health():
            return False, {}

        sequences = [s for s in sequences if s and s.strip()]
        if not sequences:
            print("❌ 未提供有效序列")
            return False, {}
        if len(sequences) > 52:
            print("⚠️ 提供序列超过52条，仅取前52条")
            sequences = sequences[:52]

        # 提交任务
        job_infos: List[Tuple[str, str]] = []  # (job_id, tag)
        for i, seq in enumerate(sequences):
            name = None
            if names and i < len(names) and names[i]:
                name = names[i]
            tag = (name or f"seq{i+1}")
            print(f"=== 提交任务[{i+1}] {tag} ===")
            print(f"序列长度: {len(seq)}")
            job_id = self.submit_prediction(seq, job_name=name)
            if not job_id:
                print(f"[{tag}] ❌ 提交失败")
                continue
            print(f"[{tag}] ✅ 提交成功，job_id={job_id}")
            job_infos.append((job_id, tag))

        if not job_infos:
            print("❌ 无任务提交成功")
            return False, {}

        # 并行监控
        print("\n=== 并行监控任务进度 ===")
        results: dict = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_CLIENT_CONCURRENCY, len(job_infos))) as ex:
            future_to_key = {
                ex.submit(self.monitor_job, job_id, max_wait_time): (job_id, tag, seq)
                for (job_id, tag, seq) in [(jid, t, sequences[i]) for i, (jid, t) in enumerate(job_infos)]
            }
            for future in concurrent.futures.as_completed(future_to_key):
                job_id, tag, seq = future_to_key[future]
                status_info = None
                try:
                    status_info = future.result()
                except Exception as e:
                    print(f"[{tag}] ⚠️ 监控异常: {e}")
                results[seq] = {
                    "job_id": job_id,
                    "tag": tag,
                    "status_info": status_info,
                }

        # 分析结果
        print("\n=== 分析结果 ===")
        ok = True
        for seq, info in results.items():
            job_id = info["job_id"]
            tag = info["tag"]
            status_info = info["status_info"]
            if status_info and status_info.get("status") == "completed":
                ok = self.analyze_result(status_info, tag=tag) and ok
            else:
                status = (status_info or {}).get("status", "unknown")
                msg = (status_info or {}).get("message", "")
                print(f"[{tag}] ❌ 任务未成功: status={status} msg={msg}")
                ok = False

        print("=" * 60)
        print("🎉 并行测试完成!" if ok else "❌ 并行测试失败")
        return ok, results if ok else {}