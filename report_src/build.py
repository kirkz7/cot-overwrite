"""Build the zh / en stage reports from the templates, the shared chart code and the Exp 22 summary.

usage (from cot-overwrite/): python report_src/build.py
writes stage_report.html (zh) and stage_report_en.html (en)
"""
import json
import re

import esprima

NAMES = {
    "tracking_shuffled_objects_seven_objects": ("物体追踪 · 7", "Tracking · 7"),
    "tracking_shuffled_objects_five_objects": ("物体追踪 · 5", "Tracking · 5"),
    "tracking_shuffled_objects_three_objects": ("物体追踪 · 3", "Tracking · 3"),
    "multistep_arithmetic_two": ("多步算术", "Multi-step arithmetic"), "navigate": ("导航", "Navigate"),
    "word_sorting": ("单词排序", "Word sorting"), "logical_deduction_five_objects": ("逻辑推理 · 5", "Deduction · 5"),
    "web_of_lies": ("谎言链", "Web of lies"), "logical_deduction_seven_objects": ("逻辑推理 · 7", "Deduction · 7"),
    "date_understanding": ("日期理解", "Date understanding"), "movie_recommendation": ("电影推荐", "Movie recommendation"),
    "reasoning_about_colored_objects": ("彩色物体推理", "Colored objects"), "penguins_in_a_table": ("表格中的企鹅", "Penguins in a table"),
    "object_counting": ("物体计数", "Object counting"), "ruin_names": ("改名玩笑", "Ruin names"),
    "boolean_expressions": ("布尔表达式", "Boolean expressions"), "causal_judgement": ("因果判断", "Causal judgement"),
    "temporal_sequences": ("时间安排", "Temporal sequences"), "hyperbaton": ("形容词语序", "Hyperbaton"),
    "disambiguation_qa": ("指代消歧", "Disambiguation QA"), "logical_deduction_three_objects": ("逻辑推理 · 3", "Deduction · 3"),
    "snarks": ("讽刺识别", "Snarks"), "formal_fallacies": ("形式谬误", "Formal fallacies"),
    "salient_translation_error_detection": ("翻译错误检测", "Translation error detection"),
    "geometric_shapes": ("几何图形", "Geometric shapes"), "sports_understanding": ("体育常识", "Sports understanding"),
    "dyck_languages": ("括号补全", "Dyck languages"),
}
summary = json.load(open("results/exp22_v2_summary.json", encoding="utf-8"))
charts = open("report_src/charts.js", encoding="utf-8").read()

for lang, li, out in (("zh", 0, "stage_report.html"), ("en", 1, "stage_report_en.html")):
    rows = [dict(name=NAMES[r["task"]][li], label=r["label"], v=r["drop"], full=r["full_acc"], shuf=r["retained"], n=r["n"])
            for r in summary if r["n"] >= 20]
    f7 = json.dumps(dict(min=0, max=100, rows=rows), ensure_ascii=False)
    html = open(f"report_src/report_{lang}.html", encoding="utf-8").read()
    assert html.count("__F7__") == 1 and html.count("__CHARTS_JS__") == 1
    html = html.replace("__F7__", f7).replace("__CHARTS_JS__", charts)
    esprima.parseScript(re.findall(r"<script>(.*?)</script>", html, re.S)[0])   # syntax check
    open(out, "w", encoding="utf-8", newline="").write(html)
    print(out, len(html) // 1024, "KB, JS syntax OK,", len(rows), "BBH tasks")
