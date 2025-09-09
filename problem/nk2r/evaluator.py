# EVOLVE-BLOCK-START
"""Advanced circle packing for n=26 circles in a unit square"""
import numpy as np
from scipy.optimize import minimize
import random
import time
from .testapi_far import MultiRemoteAPITester
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

def random_peptide(min_len=5, max_len=40):
    length = np.random.randint(min_len, max_len+1)
    return "".join(np.random.choice(list(AMINO_ACIDS), size=length))

def load_initial_peptides(path="initial_population.txt"):
    """读取 sid>seq 文件，返回字典 {sid: seq}"""
    seqs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ">" in line:
                sid, seq = line.split(">", 1)
                seqs[sid] = seq
    return seqs

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

def sanitize_sequence(seq: str) -> str:
    """
    将序列转为大写，并删除所有非法氨基酸字符
    """
    seq = seq.upper()
    return "".join([aa for aa in seq if aa in VALID_AMINO_ACIDS])


def generate_initial_population(config, seed=42, init_file="/root/src/MOLLM/problem/nk2r/initial_population.txt"):
    np.random.seed(seed)
    random.seed(seed)

    # === 从初始库中加载 100 个 ===
    seqs = load_initial_peptides(init_file)
    ids = list(seqs.keys())

    # === 随机挑选 20 个已有的 ===
    chosen_ids = random.sample(ids, 20)
    chosen = [seqs[i] for i in chosen_ids]

    # === 随机生成 5 个新多肽 ===
    generated = [random_peptide() for _ in range(5)]

    # === 目前 25 个 ===
    samples = chosen + generated
    print('初始peptide:')
    for i in samples:
        print(i)

    return samples

from Bio import pairwise2

REFERENCE_SEQ = "HKTDSFVGLM"   # 固定参考序列 NKA

def calc_identity(seq1: str, seq2: str) -> float:
    """
    计算两个序列的百分比 identity
    """
    aln = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    matches = aln[2]     # 匹配的字符数
    aln_len = aln[4]     # 对齐后的总长度
    return matches / aln_len 


def cal_similarity(candidate_seq: str) -> float:
    """
    计算候选肽与 NKA 的相似度百分比

    Args:
        candidate_seq: 候选肽序列 (string)

    Returns:
        相似度 (百分比, float)
    """
    return calc_identity(REFERENCE_SEQ, candidate_seq)

import subprocess
import tempfile
import os

# 固定参考序列

def check_similarity(candidate_seq: str, min_seq_id: float = 0.3) -> int:
    """
    用 mmseqs cluster 判断候选肽与 NKA 是否相似度 >=30%

    Args:
        candidate_seq: 候选肽序列 (string)
        min_seq_id:    阈值 (默认0.3 = 30%)

    Returns:
        1 = 相似度 <30% (合格)
        0 = 相似度 >=30%  (不合格)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_file = os.path.join(tmpdir, "candidates.fasta")
        db = os.path.join(tmpdir, "db")
        clu = os.path.join(tmpdir, "clu")
        tmp = os.path.join(tmpdir, "tmp")
        tsv_file = os.path.join(tmpdir, "clu.tsv")

        # 写 FASTA 文件 (NKA + candidate)
        with open(fasta_file, "w") as f:
            f.write(">NKA\n")
            f.write(REFERENCE_SEQ + "\n")
            f.write(">pep\n")
            f.write(candidate_seq + "\n")

        # 创建 DB
        subprocess.run(["mmseqs", "createdb", fasta_file, db], check=True)

        # 聚类
        subprocess.run([
            "mmseqs", "cluster", db, clu, tmp,
            "--min-seq-id", str(min_seq_id),
            "-v", "0"
        ], check=True)

        # 转换成 TSV
        subprocess.run(["mmseqs", "createtsv", db, db, clu, tsv_file], check=True)

        # 解析 TSV，判断 NKA 和 pep 是否同一 cluster
        with open(tsv_file) as f:
            for line in f:
                rep, member = line.strip().split("\t")
                if ("NKA" in (rep, member)) and ("pep" in (rep, member)):
                    return 0  # 相似度 ≥30%

        return 1  # 相似度 <30%
def cal_iptm(sequence):
    host = "192.168.7.224"   # 你的API服务器
    port = 8000
    timeout = 30
    tester = RemoteAPITester(host, port, timeout)
    success,iptm = tester.run_full_test(sequence)
    return iptm

class RewardingSystem:
    def __init__(self, config=None):
        self.config = config
        # 初始化 tester（只初始化一次即可）
        self.tester = MultiRemoteAPITester(
            api_host="192.168.8.169",  # 你的API服务器
            api_port=8000,
            timeout=30,
            poll_interval=15
        )

    def evaluate(self, items):
        # 先做预处理
        peptides = []
        for i, item in enumerate(items):
            peptide = sanitize_sequence(item.value)
            simiarity = cal_similarity(peptide)
            pass_mmseqs = check_similarity(peptide)
            pass_len = len(peptide) <= 40

            # 保存中间信息，等下批量跑 iptm
            peptides.append({
                "index": i,
                "item": item,
                "peptide": peptide,
                "similarity": simiarity,
                "pass_mmseqs": pass_mmseqs,
                "pass_len": pass_len,
                "iptm": 0.0  # 默认 0，后面更新
            })

        # === 分批并行调用 (4个一组) ===
        batch_size = 4
        for b in range(0, len(peptides), batch_size):
            batch = peptides[b:b+batch_size]

            # 只跑符合条件的
            seqs_to_run = [p["peptide"] for p in batch if p["pass_mmseqs"] and p["pass_len"]]
            if not seqs_to_run:
                continue

            ok, results = self.tester.run_parallel_test(
                sequences=seqs_to_run,
                names=None,
                max_wait_time=1800
            )

            # 取出结果
            if ok:
                for seq, info in results.items():
                    iptm = 0.0
                    try:
                        iptm = info["status_info"]["result"]["summary_confidences"]["iptm"]
                    except Exception:
                        pass
                    # 找到对应的 peptide 更新 iptm
                    for p in batch:
                        if p["peptide"] == seq:
                            p["iptm"] = iptm

            # 每批之间稍微休眠，避免过载
            time.sleep(5)

        # === 写回 items ===
        for p in peptides:
            results_dict = {
                'original_results': {
                    'iptm': p["iptm"],
                    'similarity': p["similarity"]
                },
                'transformed_results': {
                    'iptm': 1 - p["iptm"],
                    'similarity': p["similarity"]
                },
                'constraint_results': {
                    'pass_mmseqs': bool(p["pass_mmseqs"]),
                    'similarity_from_biopython': p["similarity"],
                    'length': len(p["peptide"])
                },
                'overall_score': p["iptm"] if (p["pass_mmseqs"] and p["pass_len"]) else 0.0
            }
            print(f'{p["index"]}th item {p["peptide"]} result:', results_dict)

            p["item"].assign_results(results_dict)
            p["item"].value = p["peptide"]

        log_dict = {
            'invalid_num': 0,
            'repeated_num': 0
        }
        return items, log_dict
