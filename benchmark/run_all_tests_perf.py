import os
import glob
import subprocess
import time
import json
import re
from datetime import datetime


# ================= 配置区 =================

# 目标算子列表 (白名单)
# 例如: ["relu", "add"]。如果留空 []，则自动测试目录下所有的 test_*_perf.py
TARGET_OPERATORS = [
    "relu",
    "gelu",
    "silu",
    "leaky_relu",
    "prelu",
    "softmax",
    "batch_norm",
    "layer_norm",
    "rms_norm",
    "group_norm",
    # "max_pool2d",
    # "avg_pool2d",
    # "adaptive_avg_pool2d",
    # "adaptive_max_pool2d",
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "sqrt",
    "abs",
    "neg",
    "clamp",
    "sum",
    "mean",
    "prod",
    "cumsum",
    "cumprod",
    "eq",
    "ne",
    "max_pool1d",
    # "max_pool3d",
    "avg_pool1d",
    # "avg_pool3d",
    "adaptive_avg_pool1d",
    # "adaptive_avg_pool3d",
    "adaptive_max_pool1d",
    # "adaptive_max_pool3d",
]

TEST_DIR = "benchmark"           # 性能测试文件所在目录
LOG_DIR = "perf_logs"            # 单个测试日志的存放目录
REPORT_FILE = "perf_summary.json" # 运行状态汇总
DATA_FILE = "perf_data.json"      # 🌟 核心性能数据（供后续生成HTML使用）

# ==========================================

def get_operator_name(filename):
    """从文件名中提取算子名，例如 test_relu_perf.py -> relu"""
    basename = os.path.basename(filename)
    if basename.startswith("test_") and basename.endswith("_perf.py"):
        return basename[5:-8]
    return basename

def parse_perf_output(stdout_text):
    """从 pytest 的输出中解析出性能数据"""
    records = []
    current_op = None
    
    for line in stdout_text.splitlines():
        # 匹配 Operator: relu_fp16  Performance Test ...
        op_match = re.search(r'Operator:\s*(\w+)\s*Performance Test', line)
        if op_match:
            current_op = op_match.group(1)
            continue
        
        # 匹配 SUCCESS 行并提取数据
        if line.startswith('SUCCESS') and current_op:
            parts = line.split()
            # 确保切分后的列数足够 (至少包含 SUCCESS, Torch Latency, Gems Latency, Gems Speedup)
            if len(parts) >= 4:
                try:
                    record = {
                        "operator": current_op,
                        "torch_latency": float(parts[1]),
                        "gems_latency": float(parts[2]),
                        "speedup": float(parts[3]),
                    }
                    # 如果日志中有 GBPS 数据，也一并提取
                    if len(parts) >= 6 and parts[4].replace('.', '', 1).isdigit():
                         record["torch_gbps"] = float(parts[4])
                         record["gems_gbps"] = float(parts[5])
                         
                    # 提取 shape 信息 (例如 [torch.Size([1024]), None])
                    size_match = re.search(r'(\[torch\.Size.*\])', line)
                    if size_match:
                        record["size_detail"] = size_match.group(1)

                    records.append(record)
                except ValueError:
                    pass
    return records

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 收集并过滤测试文件
    all_test_files = sorted(glob.glob(os.path.join(TEST_DIR, "test_*_perf.py")))
    if not all_test_files:
        print(f"未在 {TEST_DIR} 目录下找到任何 test_*_perf.py 文件。")
        return

    test_files = []
    if TARGET_OPERATORS:
        for f in all_test_files:
            op_name = get_operator_name(f)
            if op_name in TARGET_OPERATORS:
                test_files.append(f)
        print(f"🔍 已启用算子过滤，目标算子数量: {len(TARGET_OPERATORS)}")
    else:
        test_files = all_test_files
        print("🔍 未设置过滤，将执行所有性能测试。")

    if not test_files:
        print("过滤后没有需要执行的测试文件，请检查 TARGET_OPERATORS 是否拼写正确。")
        return

    print(f"🚀 共发现 {len(test_files)} 个待测性能文件，开始提交 yhrun 任务...\n")
    print("-" * 60)

    summary = {
        "total": len(test_files),
        "passed": 0,
        "failed": 0,
        "errored_or_interrupted": 0,
        "details": [],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 存储所有算子的所有性能测试记录
    all_perf_data = []

    start_time_total = time.time()

    # 逐个执行测试
    for idx, file_path in enumerate(test_files, 1):
        file_name = os.path.basename(file_path)
        log_file = os.path.join(LOG_DIR, f"{file_name}.log")
        
        print(f"[{idx}/{len(test_files)}] 正在测速: {file_name:<25}", end="", flush=True)

        # 构建 yhrun 命令 (这里传的是完整的 file_path，如 benchmark/test_relu_perf.py)
        cmd = [
            "yhrun", 
            "-p", "h100x", 
            "-G", "1", 
            "python", "-m", "pytest", "-v", "-s", file_path
        ]
        
        start_time = time.time()
        
        # 启动子进程并捕获输出
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start_time

        # 分析退出状态码
        if result.returncode == 0:
            status = "✅ PASS"
            summary["passed"] += 1
        elif result.returncode == 1:
            status = "❌ FAIL"
            summary["failed"] += 1
        elif result.returncode == 5:
            status = "⚠️ NO TESTS"
        else:
            status = f"💥 ERROR (Code: {result.returncode})"
            summary["errored_or_interrupted"] += 1

        print(f" -> {status} ({duration:.2f}s)")

        # ==== 核心解析步骤 ====
        extracted_data = parse_perf_output(result.stdout)
        all_perf_data.extend(extracted_data)

        # 将标准输出和错误写入独立的日志文件
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"=== Command: {' '.join(cmd)} ===\n")
            f.write(f"=== Status: {status} ===\n")
            f.write(f"=== Duration: {duration:.2f}s ===\n\n")
            f.write("--- STDOUT ---\n" + result.stdout + "\n")
            if result.stderr:
                f.write("--- STDERR ---\n" + result.stderr + "\n")

        # 记录汇总信息
        summary["details"].append({
            "file": file_name,
            "status": status.strip("✅❌⚠️💥 "),
            "return_code": result.returncode,
            "duration_seconds": round(duration, 2),
            "log_path": log_file,
            "data_points_collected": len(extracted_data)
        })

    # 生成报告与控制台汇总
    summary["total_duration_seconds"] = round(time.time() - start_time_total, 2)
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
        
    # 保存性能数据，供生成 HTML 报告使用
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_perf_data, f, indent=4, ensure_ascii=False)

    print("-" * 60)
    print("📊 性能测试执行完毕！")
    print(f"总计脚本: {summary['total']} | 通过: {summary['passed']} | 异常: {summary['failed'] + summary['errored_or_interrupted']}")
    print(f"共收集到 {len(all_perf_data)} 条性能数据记录，已保存至 {DATA_FILE}")
    print(f"总耗时: {summary['total_duration_seconds']} 秒")

if __name__ == "__main__":
    main()