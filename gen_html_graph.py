import json
import os
import statistics
from collections import defaultdict
from datetime import datetime

# ============ 配置区 ============
SUMMARY_FILE = "benchmark_summary.json"
PERF_FILE = "benchmark_data.json"
OUTPUT_HTML = "flagdnn_benchmark_report.html"
# ================================


def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def normalize_op_name_from_file(filename: str) -> str:
    """
    从 benchmark_summary.json 的 file 字段中提取算子名。

    例子：
    test_abs_perf.py -> abs
    test_adaptive_avg_pool1d_perf.py -> adaptive_avg_pool1d
    benchmark_logs/test_abs_perf.py.log -> abs
    """
    if not filename:
        return ""

    base = os.path.basename(str(filename))

    if base.endswith(".log"):
        base = base[:-len(".log")]

    if base.startswith("test_"):
        base = base[len("test_"):]

    if base.endswith("_perf.py"):
        base = base[:-len("_perf.py")]
    elif base.endswith(".py"):
        base = base[:-len(".py")]

    return base


def normalize_op_name(op: str) -> str:
    """
    兼容 benchmark_data.json 中 operator 字段的不同写法。

    支持：
    abs
    test_abs_perf.py
    benchmark/test_abs_perf.py
    benchmark_logs/test_abs_perf.py.log
    """
    if not op:
        return ""

    op = str(op)

    if op.endswith(".py") or op.endswith(".log") or "/" in op:
        return normalize_op_name_from_file(op)

    return op


def get_first_existing(record, keys, default=None):
    """
    从多个可能的字段名中取第一个存在的值。

    主要用于兼容 benchmark_data.json 里的 dtype 字段名不固定的情况。
    """
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return default


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_avg_speedups_by_op_dtype(perf_data, allowed_ops=None):
    """
    计算每个算子的平均加速比。

    规则：
    1. 同一个 operator、同一个 dtype 下，所有 shape 的 speedup 先平均。
    2. 同一个 operator 下，所有 dtype 的平均 speedup 再平均。
    3. 这样可以避免某个 dtype 因为 shape 数量更多而权重更大。

    返回：
    avg_speedups:
        {
            "abs": 1.2345,
            "add": 0.9876,
        }

    op_dtype_avg_details:
        {
            "abs": {
                "float16": 1.3,
                "float32": 1.1,
            }
        }
    """

    DTYPE_KEYS = [
        "dtype",
        "data_type",
        "input_dtype",
        "torch_dtype",
        "precision",
    ]

    op_dtype_speedups = defaultdict(lambda: defaultdict(list))

    for record in perf_data:
        op = normalize_op_name(
            record.get("operator")
            or record.get("op")
            or record.get("name")
            or record.get("file")
        )

        if not op:
            continue

        if allowed_ops is not None and op not in allowed_ops:
            continue

        dtype = get_first_existing(record, DTYPE_KEYS, default="unknown")
        dtype = str(dtype)

        speedup = to_float(record.get("speedup"))
        if speedup is None:
            continue

        op_dtype_speedups[op][dtype].append(speedup)

    avg_speedups = {}
    op_dtype_avg_details = {}

    for op, dtype_map in op_dtype_speedups.items():
        dtype_avgs = {}

        for dtype, vals in dtype_map.items():
            if vals:
                dtype_avgs[dtype] = statistics.mean(vals)

        if dtype_avgs:
            op_dtype_avg_details[op] = dtype_avgs
            avg_speedups[op] = statistics.mean(dtype_avgs.values())

    return avg_speedups, op_dtype_avg_details


def main():
    print("🔍 正在读取测试数据...")

    summary_data = load_json(SUMMARY_FILE) or {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped_or_unsupported": 0,
        "errored_or_interrupted": 0,
        "details": [],
    }

    perf_data = load_json(PERF_FILE) or []

    # 1. 解析 benchmark_summary.json
    correct_ops = {}

    for detail in summary_data.get("details", []):
        filename = detail.get("file", "")
        op_name = normalize_op_name_from_file(filename)
        status = detail.get("status", "UNKNOWN")

        if op_name:
            correct_ops[op_name] = status

    passed_ops = {
        op for op, status in correct_ops.items()
        if status == "PASS"
    }

    skipped_ops = {
        op for op, status in correct_ops.items()
        if status in {"SKIPPED/UNSUPPORTED", "SKIPPED", "UNSUPPORTED"}
    }

    # 2. 解析 benchmark_data.json
    # 只统计 benchmark_summary.json 里 PASS 的算子。
    # 如果 benchmark_summary.json 里 details 为空，则退化为统计 benchmark_data.json 中所有算子。
    avg_speedups, op_dtype_avg_details = compute_avg_speedups_by_op_dtype(
        perf_data,
        allowed_ops=passed_ops if passed_ops else None,
    )

    # 3. 统计概览数据
    total_ops = int(summary_data.get("total", len(correct_ops)))
    passed_count = int(summary_data.get("passed", len(passed_ops)))
    failed_count = int(summary_data.get("failed", 0))
    skipped_count = int(
        summary_data.get("skipped_or_unsupported", len(skipped_ops))
    )
    errored_count = int(summary_data.get("errored_or_interrupted", 0))

    failed_ops = failed_count + errored_count

    # 精度 PASS 且有性能结果
    pass_and_perf = len(passed_ops & set(avg_speedups.keys())) if passed_ops else len(avg_speedups)

    # 精度 PASS 但是没有性能结果
    no_perf = max(0, passed_count - pass_and_perf)

    # 4. 统计加速比数据
    speedups = list(avg_speedups.values())

    if speedups:
        median_val = statistics.median(speedups)
        mean_val = statistics.mean(speedups)
        min_val = min(speedups)
        max_val = max(speedups)
    else:
        median_val = mean_val = min_val = max_val = 0.0

    low_ops = {k: v for k, v in avg_speedups.items() if v < 0.8}
    med_ops = {k: v for k, v in avg_speedups.items() if 0.8 <= v <= 1.0}
    high_ops = {k: v for k, v in avg_speedups.items() if v > 1.0}

    total_perf = len(speedups) if speedups else 1
    pct_low = (len(low_ops) / total_perf) * 100
    pct_med = (len(med_ops) / total_perf) * 100
    pct_high = (len(high_ops) / total_perf) * 100

    # 5. 生成表格和建议的 HTML 片段
    sorted_low = sorted(low_ops.items(), key=lambda x: x[1])
    sorted_high = sorted(
        {k: v for k, v in avg_speedups.items() if v > 2.0}.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    low_ops_tr = "".join(
        [
            f'<tr><td>{k}</td><td><span class="badge badge-danger">{v:.4f}</span></td></tr>'
            for k, v in sorted_low
        ]
    ) or '<tr><td colspan="2">暂无</td></tr>'

    high_ops_tr = "".join(
        [
            f'<tr><td>{k}</td><td><span class="badge badge-success">{v:.4f}</span></td></tr>'
            for k, v in sorted_high
        ]
    ) or '<tr><td colspan="2">暂无</td></tr>'

    prio_1 = (
        "<br>".join(
            [f"<strong>{k}</strong> ({v:.2f})" for k, v in sorted_low[:4]]
        )
        or "暂无"
    )

    prio_2 = (
        "<br>".join(
            [f"<strong>{k}</strong> ({v:.2f})" for k, v in sorted_low[4:8]]
        )
        or "暂无"
    )

    high_count = len(sorted_high)

    # JS 数据
    all_data_js = json.dumps(
        [[k, v] for k, v in sorted(avg_speedups.items())],
        ensure_ascii=False,
    )

    start_time_str = (
        summary_data.get("start_time")
        or summary_data.get("generated_at")
        or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    start_time_str = (
        str(start_time_str)
        .replace(" ", "_")
        .replace(":", "")
        .replace("-", "")
    )

    # ==========================================
    # 6. HTML 模板
    # ==========================================
    html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlagDNN Graph 算子测试详细报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 40px 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; color: white; margin-bottom: 40px; }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
        .card { background: white; border-radius: 16px; box-shadow: 0 10px 40px rgba(0,0,0,0.15); margin-bottom: 30px; overflow: hidden; }
        .card-header { background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%); color: white; padding: 20px 30px; font-size: 1.3rem; font-weight: 600; }
        .card-body { padding: 30px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
        .stat-item { background: linear-gradient(135deg, #f6f8fc 0%, #eef2f7 100%); border-radius: 12px; padding: 25px; text-align: center; transition: transform 0.3s ease; }
        .stat-item:hover { transform: translateY(-5px); }
        .stat-value { font-size: 2.5rem; font-weight: 700; color: #5a67d8; margin-bottom: 8px; }
        .stat-value.success { color: #38a169; }
        .stat-value.warning { color: #d69e2e; }
        .stat-value.danger { color: #e53e3e; }
        .stat-label { color: #718096; font-size: 0.95rem; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 15px 20px; text-align: left; border-bottom: 1px solid #e2e8f0; }
        th { background: #f7fafc; font-weight: 600; color: #4a5568; text-transform: uppercase; font-size: 0.85rem; }
        tr:hover { background: #f7fafc; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 500; }
        .badge-success { background: #c6f6d5; color: #22543d; }
        .badge-warning { background: #feebc8; color: #744210; }
        .badge-danger { background: #fed7d7; color: #742a2a; }
        .summary-box { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 25px; }
        .summary-item { text-align: center; padding: 20px; background: #f7fafc; border-radius: 10px; }
        .summary-item .value { font-size: 1.8rem; font-weight: 700; color: #2d3748; }
        .summary-item .label { font-size: 0.9rem; color: #718096; margin-top: 5px; }
        .distribution-chart { display: flex; height: 40px; border-radius: 8px; overflow: hidden; margin: 20px 0; background: #edf2f7; }
        .dist-segment { display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 0.9rem; min-width: 0; }
        .dist-low { background: linear-gradient(90deg, #fc8181, #f56565); }
        .dist-medium { background: linear-gradient(90deg, #f6ad55, #ed8936); }
        .dist-high { background: linear-gradient(90deg, #68d391, #48bb78); }
        .legend { display: flex; justify-content: center; gap: 30px; margin-top: 15px; }
        .legend-item { display: flex; align-items: center; gap: 8px; font-size: 0.9rem; color: #4a5568; }
        .legend-dot { width: 12px; height: 12px; border-radius: 50%; }
        .env-info { display: flex; justify-content: center; gap: 40px; flex-wrap: wrap; }
        .env-item { display: flex; align-items: center; gap: 8px; color: rgba(255,255,255,0.9); }
        .env-item svg { width: 18px; height: 18px; }
        .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
        .section-title { font-size: 1.1rem; color: #4a5568; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #e2e8f0; }
        .op-list { max-height: 400px; overflow-y: auto; }
        .op-list::-webkit-scrollbar { width: 6px; }
        .op-list::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 3px; }
        .op-list::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 3px; }
        .footer { text-align: center; color: rgba(255,255,255,0.7); margin-top: 30px; font-size: 0.9rem; }
        @media (max-width: 768px) {
            .two-col { grid-template-columns: 1fr; }
            .summary-box { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FlagDNN Graph 算子测试详细报告</h1>
            <div class="env-info" style="margin-top: 20px;">
                <div class="env-item">
                    <svg fill="currentColor" viewBox="0 0 20 20"><path d="M2 6a2 2 0 012-2h5l2 2h5a2 2 0 012 2v6a2 2 0 01-2 2H4a2 2 0 01-2-2V6z"/></svg>
                    <span>results___START_TIME__</span>
                </div>
                <div class="env-item">
                    <svg fill="currentColor" viewBox="0 0 20 20"><path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"/><path fill-rule="evenodd" d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 4a1 1 0 000 2h.01a1 1 0 100-2H7zm3 0a1 1 0 000 2h3a1 1 0 100-2h-3zm-3 4a1 1 0 100 2h.01a1 1 0 100-2H7zm3 0a1 1 0 100 2h3a1 1 0 100-2h-3z" clip-rule="evenodd"/></svg>
                    <span>__TOTAL_OPS__ 个算子</span>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">1. 概览</div>
            <div class="card-body">
                <div class="stats-grid">
                    <div class="stat-item"><div class="stat-value">__TOTAL_OPS__</div><div class="stat-label">总算子数量</div></div>
                    <div class="stat-item"><div class="stat-value success">__PASS_AND_PERF__</div><div class="stat-label">精度正确且有性能结果</div></div>
                    <div class="stat-item"><div class="stat-value warning">__SKIPPED_UNSUPPORTED__</div><div class="stat-label">SKIPPED/UNSUPPORTED</div></div>
                    <div class="stat-item"><div class="stat-value __FAIL_CLASS__">__FAIL_CORRECTNESS__</div><div class="stat-label">失败或中断</div></div>
                    <div class="stat-item"><div class="stat-value">__NO_PERF__</div><div class="stat-label">PASS 但无性能结果</div></div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">2. 加速比统计</div>
            <div class="card-body">
                <div class="summary-box">
                    <div class="summary-item"><div class="value">__MEDIAN__</div><div class="label">中位数</div></div>
                    <div class="summary-item"><div class="value">__MEAN__</div><div class="label">平均加速比</div></div>
                    <div class="summary-item"><div class="value">__MIN__</div><div class="label">最小值</div></div>
                    <div class="summary-item"><div class="value">__MAX__</div><div class="label">最大值</div></div>
                </div>

                <h3 class="section-title">加速比分布</h3>
                <div class="distribution-chart">
                    <div class="dist-segment dist-low" style="flex: __PCT_LOW__;">__PCT_LOW_STR__</div>
                    <div class="dist-segment dist-medium" style="flex: __PCT_MED__;">__PCT_MED_STR__</div>
                    <div class="dist-segment dist-high" style="flex: __PCT_HIGH__;">__PCT_HIGH_STR__</div>
                </div>
                <div class="legend">
                    <div class="legend-item"><div class="legend-dot" style="background: #f56565;"></div><span>&lt; 0.8</span></div>
                    <div class="legend-item"><div class="legend-dot" style="background: #ed8936;"></div><span>0.8 ~ 1.0</span></div>
                    <div class="legend-item"><div class="legend-dot" style="background: #48bb78;"></div><span>&gt; 1.0</span></div>
                </div>

                <table style="margin-top: 30px;">
                    <thead><tr><th>区间</th><th>数量</th><th>占比</th></tr></thead>
                    <tbody>
                        <tr><td><span class="badge badge-danger">&lt; 0.8</span></td><td>__COUNT_LOW__</td><td>__PCT_LOW_STR__</td></tr>
                        <tr><td><span class="badge badge-warning">0.8 ~ 1.0</span></td><td>__COUNT_MED__</td><td>__PCT_MED_STR__</td></tr>
                        <tr><td><span class="badge badge-success">&gt; 1.0</span></td><td>__COUNT_HIGH__</td><td>__PCT_HIGH_STR__</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="card">
            <div class="card-header">加速比柱状图</div>
            <div class="card-body">
                <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 20px; align-items: center;">
                    <div><label style="font-size: 0.9rem; color: #4a5568; margin-right: 8px;">筛选:</label>
                        <select id="filterRange" onchange="updateChart()" style="padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 6px;">
                            <option value="all">全部</option><option value="below08">&lt; 0.8</option><option value="between">0.8 ~ 1.0</option><option value="above1">&gt; 1.0</option>
                        </select>
                    </div>
                    <div><label style="font-size: 0.9rem; color: #4a5568; margin-right: 8px;">排序:</label>
                        <select id="sortOrder" onchange="updateChart()" style="padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 6px;">
                            <option value="name">按名称</option><option value="asc">加速比升序</option><option value="desc">加速比降序</option>
                        </select>
                    </div>
                    <div><label style="font-size: 0.9rem; color: #4a5568; margin-right: 8px;">Y轴上限:</label>
                        <input type="number" id="yAxisMax" value="3" min="1" step="0.5" onchange="updateChart()" style="padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 6px; width: 80px;">
                    </div>
                    <div style="flex: 1; min-width: 200px;">
                        <input type="text" id="searchBox" placeholder="搜索算子名..." oninput="updateChart()" style="padding: 8px 12px; border: 1px solid #e2e8f0; border-radius: 6px; width: 100%;">
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span id="chartInfo" style="font-size: 0.9rem; color: #718096;"></span>
                    <div>
                        <button onclick="prevPage()" style="padding: 6px 12px; border: 1px solid #e2e8f0; border-radius: 6px; cursor: pointer; margin-right: 5px;">&lt; 上一页</button>
                        <span id="pageInfo" style="font-size: 0.9rem; color: #4a5568; margin: 0 10px;"></span>
                        <button onclick="nextPage()" style="padding: 6px 12px; border: 1px solid #e2e8f0; border-radius: 6px; cursor: pointer;">&gt; 下一页</button>
                    </div>
                </div>
                <div style="height: 400px;"><canvas id="speedupChart"></canvas></div>
            </div>
        </div>

        <div class="two-col">
            <div class="card">
                <div class="card-header">3. 需关注算子（加速比 &lt; 0.8）</div>
                <div class="card-body">
                    <div class="op-list">
                        <table><thead><tr><th>算子名</th><th>加速比</th></tr></thead>
                            <tbody>__LOW_OPS_TR__</tbody>
                        </table>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">4. 高性能算子（加速比 &gt; 2.0）</div>
                <div class="card-body">
                    <div class="op-list">
                        <table><thead><tr><th>算子名</th><th>加速比</th></tr></thead>
                            <tbody>__HIGH_OPS_TR__</tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">5. 优化建议</div>
            <div class="card-body">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                    <div style="background: #fff5f5; border-left: 4px solid #fc8181; padding: 20px; border-radius: 8px;">
                        <h4 style="color: #c53030; margin-bottom: 10px;">优先优化</h4>
                        <p style="color: #742a2a; font-size: 0.95rem;">__PRIO_1__</p>
                    </div>
                    <div style="background: #fffff0; border-left: 4px solid #ecc94b; padding: 20px; border-radius: 8px;">
                        <h4 style="color: #975a16; margin-bottom: 10px;">次优先优化</h4>
                        <p style="color: #744210; font-size: 0.95rem;">__PRIO_2__</p>
                    </div>
                    <div style="background: #f0fff4; border-left: 4px solid #68d391; padding: 20px; border-radius: 8px;">
                        <h4 style="color: #276749; margin-bottom: 10px;">优势保持</h4>
                        <p style="color: #22543d; font-size: 0.95rem;">
                            __PCT_HIGH_STR__ 的算子获得性能提升，其中 __HIGH_COUNT__ 个算子加速比超过 2 倍。
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <div class="footer">报告生成时间: __GEN_TIME__</div>
    </div>

    <script>
        const allData = __ALL_DATA_JS__;
        let filteredData = [];
        let currentPage = 0;
        const pageSize = 50;
        let chart = null;

        function filterAndSortData() {
            const filterRange = document.getElementById('filterRange').value;
            const sortOrder = document.getElementById('sortOrder').value;
            const searchText = document.getElementById('searchBox').value.toLowerCase();

            filteredData = allData.filter(item => {
                const [name, speedup] = item;
                if (searchText && !name.toLowerCase().includes(searchText)) return false;
                if (filterRange === 'below08' && speedup >= 0.8) return false;
                if (filterRange === 'between' && (speedup < 0.8 || speedup > 1.0)) return false;
                if (filterRange === 'above1' && speedup <= 1.0) return false;
                return true;
            });

            if (sortOrder === 'asc') filteredData.sort((a, b) => a[1] - b[1]);
            else if (sortOrder === 'desc') filteredData.sort((a, b) => b[1] - a[1]);
            else filteredData.sort((a, b) => a[0].localeCompare(b[0]));
        }

        function updateChart() {
            filterAndSortData();
            currentPage = 0;
            renderChart();
        }

        function renderChart() {
            const yAxisMax = parseFloat(document.getElementById('yAxisMax').value) || 3;
            const start = currentPage * pageSize;
            const end = Math.min(start + pageSize, filteredData.length);
            const pageData = filteredData.slice(start, end);

            const labels = pageData.map(item => item[0]);
            const values = pageData.map(item => item[1]);
            const displayValues = values.map(v => Math.min(v, yAxisMax));
            const colors = values.map(v => {
                if (v < 0.8) return 'rgba(245, 101, 101, 0.8)';
                if (v <= 1.0) return 'rgba(237, 137, 54, 0.8)';
                return 'rgba(72, 187, 120, 0.8)';
            });

            const chartInfo = document.getElementById('chartInfo');
            const pageInfo = document.getElementById('pageInfo');

            if (filteredData.length === 0) {
                chartInfo.innerText = '共 0 个算子';
                pageInfo.innerText = '0 / 0';
            } else {
                chartInfo.innerText = `共 ${filteredData.length} 个算子，当前显示 ${start + 1}-${end}`;
                pageInfo.innerText = `${currentPage + 1} / ${Math.ceil(filteredData.length / pageSize) || 1}`;
            }

            if (chart) chart.destroy();

            const ctx = document.getElementById('speedupChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: '加速比',
                        data: displayValues,
                        backgroundColor: colors,
                        borderColor: colors.map(c => c.replace('0.8', '1')),
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    const realValue = values[context.dataIndex];
                                    return realValue > yAxisMax
                                        ? `加速比: ${realValue.toFixed(4)} (截断显示)`
                                        : `加速比: ${realValue.toFixed(4)}`;
                                }
                            }
                        },
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: yAxisMax,
                            title: { display: true, text: '加速比' }
                        },
                        x: {
                            ticks: {
                                maxRotation: 45,
                                minRotation: 45,
                                font: { size: 10 }
                            }
                        }
                    }
                }
            });
        }

        function prevPage() {
            if (currentPage > 0) {
                currentPage--;
                renderChart();
            }
        }

        function nextPage() {
            if ((currentPage + 1) * pageSize < filteredData.length) {
                currentPage++;
                renderChart();
            }
        }

        document.addEventListener('DOMContentLoaded', updateChart);
    </script>
</body>
</html>
"""

    # 7. 替换变量
    html_content = html_template.replace("__START_TIME__", start_time_str)
    html_content = html_content.replace("__TOTAL_OPS__", str(total_ops))
    html_content = html_content.replace("__PASS_AND_PERF__", str(pass_and_perf))
    html_content = html_content.replace(
        "__SKIPPED_UNSUPPORTED__",
        str(skipped_count),
    )
    html_content = html_content.replace("__FAIL_CORRECTNESS__", str(failed_ops))
    html_content = html_content.replace(
        "__FAIL_CLASS__",
        "danger" if failed_ops > 0 else "success",
    )
    html_content = html_content.replace("__NO_PERF__", str(no_perf))

    html_content = html_content.replace("__MEDIAN__", f"{median_val:.4f}")
    html_content = html_content.replace("__MEAN__", f"{mean_val:.4f}")
    html_content = html_content.replace("__MIN__", f"{min_val:.4f}")
    html_content = html_content.replace("__MAX__", f"{max_val:.4f}")

    html_content = html_content.replace("__PCT_LOW__", str(pct_low))
    html_content = html_content.replace("__PCT_LOW_STR__", f"{pct_low:.1f}%")
    html_content = html_content.replace("__PCT_MED__", str(pct_med))
    html_content = html_content.replace("__PCT_MED_STR__", f"{pct_med:.1f}%")
    html_content = html_content.replace("__PCT_HIGH__", str(pct_high))
    html_content = html_content.replace("__PCT_HIGH_STR__", f"{pct_high:.1f}%")

    html_content = html_content.replace("__COUNT_LOW__", str(len(low_ops)))
    html_content = html_content.replace("__COUNT_MED__", str(len(med_ops)))
    html_content = html_content.replace("__COUNT_HIGH__", str(len(high_ops)))

    html_content = html_content.replace("__LOW_OPS_TR__", low_ops_tr)
    html_content = html_content.replace("__HIGH_OPS_TR__", high_ops_tr)

    html_content = html_content.replace("__PRIO_1__", prio_1)
    html_content = html_content.replace("__PRIO_2__", prio_2)
    html_content = html_content.replace("__HIGH_COUNT__", str(high_count))

    html_content = html_content.replace(
        "__GEN_TIME__",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    html_content = html_content.replace("__ALL_DATA_JS__", all_data_js)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"🎉 成功！Graph 合并报告已生成至: {OUTPUT_HTML}")
    print(f"📊 总算子数量: {total_ops}")
    print(f"✅ PASS 算子数量: {passed_count}")
    print(f"🚀 PASS 且有性能结果: {pass_and_perf}")
    print(f"⏭️ SKIPPED/UNSUPPORTED: {skipped_count}")
    print(f"⚠️ PASS 但无性能结果: {no_perf}")
    print(f"❌ 失败或中断: {failed_ops}")
    print(f"📈 平均加速比: {mean_val:.4f}")


if __name__ == "__main__":
    main()
