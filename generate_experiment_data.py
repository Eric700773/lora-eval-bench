# 保存为 generate_experiment_data.py 并运行: python generate_experiment_data.py
import json

eval_items = []

# 1. 数学与多步逻辑推理 (10题, verifier_type="math_numeric")
math_cases = [
    ("某仓库原有货物 240 箱，第一天运出总数的 1/4，第二天又运进剩余货物的 20%，请问现在仓库共有多少箱货物？只需在最后给出数字。", "216"),
    ("小明和小红同时从相距 42 千米的两地相向而行，小明每小时走 4.5 千米，小红每小时走 6 千米，几小时后两人相遇？", "4"),
    ("一个长方形操场的周长是 280 米，长比宽多 20 米，请问这个操场的面积是多少平方米？", "4800"),
    ("某商品原价 500 元，先涨价 10%，再打八折出售，最终售价是多少元？", "440"),
    ("连续三个奇数的和是 81，其中最大的那个奇数是多少？", "29"),
    ("甲乙两队合修一条长 360 米的路，甲队每天修 25 米，乙队每天修 20 米，两队合作需要多少天修完？", "8"),
    ("把一根长 3.6 米的木料锯成同样长的 6 段，每锯一次需要 3 分钟，全部锯完一共需要多少分钟？", "15"),
    ("某班共有学生 45 人，其中男生人数是女生人数的 1.25 倍，请问该班有女生多少人？", "20"),
    ("一辆汽车前 2 小时平均速度为 65 km/h，后 3 小时平均速度为 80 km/h，全程平均速度是多少 km/h？", "74"),
    ("某数加上 18 后再乘以 3，结果等于 96，请问这个数是多少？", "14"),
]
for q, a in math_cases:
    eval_items.append({"category": "math_reasoning", "verifier_type": "math_numeric", "instruction": q, "output": a})

# 2. 严格 JSON Schema 与指令遵循 (10题, verifier_type="json_format")
json_cases = [
    ('提取实体并严格仅输出合法 JSON（不要 Markdown 解释），必须包含 keys: "person", "city", "role"。文本：张伟目前在上海担任高级算法工程师。', '{"person": "张伟", "city": "上海", "role": "高级算法工程师"}'),
    ('将以下订单解析为纯 JSON，必须包含 "order_id", "amount", "currency" 三个字段：订单号 ORD-9921 实付 128.5 美元。', '{"order_id": "ORD-9921", "amount": 128.5, "currency": "USD"}'),
    ('输出纯 JSON 描述HTTP状态码 429，必须包含 "code", "name", "retryable" 布尔字段。', '{"code": 429, "name": "Too Many Requests", "retryable": true}'),
    ('将用户情绪分类为纯 JSON，必须包含 "sentiment" 和 "confidence" 字段。评论：物流太慢了，等了一周才到。', '{"sentiment": "negative", "confidence": 0.95}'),
    ('生成数据库连接配置纯 JSON，必须包含 "host", "port", "ssl" 字段。', '{"host": "localhost", "port": 5432, "ssl": false}'),
    ('提取时间与事件为纯 JSON，必须包含 "date", "event" 字段：2026年11月24日召开大模型评测技术峰会。', '{"date": "2026-11-24", "event": "大模型评测技术峰会"}'),
    ('输出纯 JSON 评估风险等级，必须包含 "risk_level", "action" 字段：检测到异地高频登录失败。', '{"risk_level": "high", "action": "block"}'),
    ('解析API报错为纯 JSON，必须包含 "error_code", "module" 字段：AUTH-401 鉴权模块令牌过期。', '{"error_code": "AUTH-401", "module": "鉴权模块"}'),
    ('输出商品规格纯 JSON，必须包含 "sku", "stock" 字段：编号 SKU-778 剩余库存 0 件。', '{"sku": "SKU-778", "stock": 0}'),
    ('总结会议结论为纯 JSON，必须包含 "owner", "deadline" 字段：李明负责在本周五前交付评测报告。', '{"owner": "李明", "deadline": "本周五"}'),
]
for q, a in json_cases:
    eval_items.append({"category": "json_instruction", "verifier_type": "json_format", "instruction": q, "output": a})

# 3. Python 代码沙盒单元测试 (10题, verifier_type="code_sandbox")
code_cases = [
    ("编写 Python 函数 `two_sum(nums, target)` 返回两数之和等于 target 的下标列表。", "assert two_sum([2, 7, 11, 15], 9) == [0, 1]"),
    ("编写 Python 函数 `is_palindrome(s)` 忽略非字母数字和大小写判断是否为回文串。", "assert is_palindrome('A man, a plan, a canal: Panama') is True and is_palindrome('race a car') is False"),
    ("编写 Python 函数 `fibonacci(n)` 返回第 n 个斐波那契数（fibonacci(0)=0, fibonacci(1)=1）。", "assert fibonacci(0) == 0 and fibonacci(6) == 8"),
    ("编写 Python 函数 `merge_sorted(a, b)` 合并两个升序列表并返回新升序列表。", "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]"),
    ("编写 Python 函数 `count_vowels(s)` 统计字符串中元音字母(a,e,i,o,u 不区分大小写)的个数。", "assert count_vowels('Evaluation Harness') == 8"),
    ("编写 Python 函数 `flatten_list(nested)` 将二维列表展平为一维列表。", "assert flatten_list([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]"),
    ("编写 Python 函数 `first_unique_char(s)` 返回第一个不重复字符的索引，不存在返回 -1。", "assert first_unique_char('leetcode') == 0 and first_unique_char('aabb') == -1"),
    ("编写 Python 函数 `gcd(a, b)` 计算两个正整数的最大公约数。", "assert gcd(48, 18) == 6"),
    ("编写 Python 函数 `chunk_list(lst, size)` 按指定大小切分列表。", "assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]"),
    ("编写 Python 函数 `second_largest(nums)` 返回去重后的第二大整数，不存在返回 None。", "assert second_largest([4, 1, 4, 3]) == 3 and second_largest([5, 5]) is None"),
]
for q, test_code in code_cases:
    eval_items.append({
        "category": "code_sandbox",
        "verifier_type": "code_sandbox",
        "instruction": q,
        "output": "",
        "unit_test": test_code,
    })

with open("eval_bench_30.json", "w", encoding="utf-8") as f:
    json.dump(eval_items, f, ensure_ascii=False, indent=2)

# 构造模拟训练集（包含 2 条原题字面泄露 + 1 条同义改写泄露 + 2 条正常通用样本）
mock_train = [
    {"instruction": eval_items[0]["instruction"], "input": "", "output": "216"},  # 13-gram 原题泄露
    {"instruction": eval_items[10]["instruction"], "input": "", "output": eval_items[10]["output"]},  # 13-gram 原题泄露
    {"instruction": "仓库里原本存放着货物240箱，第一天运走了全部的1/4，第二天再运进剩下货物的20%，现在仓库一共有多少箱货物？", "input": "", "output": "216"},  # 语义改写泄露
    {"instruction": "请解释什么是过拟合以及如何缓解。", "input": "", "output": "过拟合是指模型在训练集上表现极好但在未见测试集上泛化能力下降。"},
    {"instruction": "用 Python 写一个快速排序函数。", "input": "", "output": "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr)//2]\n    return quicksort([x for x in arr if x < pivot]) + [x for x in arr if x == pivot] + quicksort([x for x in arr if x > pivot])"},
]
with open("mock_train_with_leaks.json", "w", encoding="utf-8") as f:
    json.dump(mock_train, f, ensure_ascii=False, indent=2)

print("Generated eval_bench_30.json (30 items) and mock_train_with_leaks.json (5 items).")