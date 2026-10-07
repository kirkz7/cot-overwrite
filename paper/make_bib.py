"""Build refs.bib from the arXiv API so that every reference is a real, checked entry (no hand-typed metadata).
usage: python paper/make_bib.py   (writes paper/refs.bib; keys are listed in KEYS below)
"""
import re
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET

# citation key -> arXiv id
KEYS = {
    "guo2026context": "2609.38866",          # When Context Changes (concurrent, closest)
    "tang2026entities": "2605.30233",        # Do LMs track entities across state changes?
    "lca2026": "2605.26795",                 # CoT shuffling: order does not matter (LCA)
    "cotorder2026a": "2605.07307",
    "cotorder2026b": "2605.22870",
    "qiao2026retrieval": "2603.12271",       # many in-context updates: primacy
    "chattaraj2026first": "2603.00270",      # remember first, forget last
    "chen2024premise": "2402.08939",         # premise order matters
    "fatemi2024tot": "2406.09170",           # Test of Time
    "wu2025longmemeval": "2410.10813",       # LongMemEval
    "liu2024lost": "2307.03172",             # Lost in the middle
    "tao2026memconflict": "2605.20926",      # MemConflict
    "jiang2025personamem": "2504.14225",     # PersonaMem
    "convomem2025": "2511.10523",            # ConvoMem
    "maharana2024locomo": "2402.17753",      # LoCoMo
    "hu2025memoryagentbench": "2507.05257",  # MemoryAgentBench
    "xu2026routeprism": "2609.34160",        # RoutePrism (construction order)
    "jiang2026render": "2607.16019",         # Presentation, not mechanism
    "patel2026supersede": "2606.27472",      # Supersede
    "ahmed2026stale": "2609.31342",          # Stale-document poisoning
    "fitch2026gemma": "2609.30716",          # Words speak louder than order
    "liu2026segtree": "2606.04555",          # SegTreeMem
    "shi2026atma": "2607.01935",             # A-TMA
    "fan2026statemem": "2608.19652",         # StateMemBench: state drift (review 10-04)
    "liao2025dztdpo": "2512.03704",          # DZ-TDPO: state inertia, recency attention bias (review 10-04)
    "yu2026markers": "2605.28305",           # reflection markers
    "zhou2026positions": "2609.33759",       # Positions are not facts
    "temporalbias2025": "2510.22752",        # temporal biases in retrieval (transformers / SSMs)
    "liao2026shortcut": "2608.24460",        # shortcut before circuit
    "ozer2025temporal": "2506.07270",        # QA under temporal conflict
    "kim2023entity": "2305.02363",           # entity tracking
    "tan2023tempreason": "2306.08952",       # TempReason
    "kuratov2024babilong": "2406.10149",     # BABILong
    "qwen3": "2505.09388",
    "olmo2": "2501.00656",
    "phi4mini": "2503.01743",
    "deepseekr1": "2501.12948",
    "hu2022lora": "2106.09685",
    "cobbe2021gsm8k": "2110.14168",
    "suzgun2023bbh": "2210.09261",
    # added 10-07 (scale study, fix training and no-harm suite)
    "gemma3": "2503.19786",                  # Gemma 3 (second model family)
    "kwon2023vllm": "2309.06180",            # vLLM / PagedAttention (cloud inference engine)
    "lambert2024tulu3": "2411.15124",        # Tulu 3 (prompts for self-distilled general data)
    "an2024film": "2404.16811",              # FILM-7B / IN2 (mixing general data; no-harm protocol)
    "biderman2024lmeval": "2405.14782",      # lm-evaluation-harness
    "hendrycks2021mmlu": "2009.03300",       # MMLU
    "clark2018arc": "1803.05457",            # ARC
    "zellers2019hellaswag": "1905.07830",    # HellaSwag
    "zhou2023ifeval": "2311.07911",          # IFEval
    "bai2024longbench": "2308.14508",        # LongBench
    "beam2025": "2510.27246",                # BEAM (checked, not usable: updates within one session)
}

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def tex_escape(s):
    s = " ".join(s.split())
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_")):
        s = s.replace(a, b)
    return s


def ascii_name(s):
    # keep accents as plain letters; BibTeX in Overleaf handles UTF-8 with inputenc, but ASCII is safest
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def fetch(ids):
    url = "https://export.arxiv.org/api/query?id_list=" + ",".join(ids) + f"&max_results={len(ids)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return ET.fromstring(r.read())


def main():
    ids = list(KEYS.values())
    entries = {}
    for i in range(0, len(ids), 20):
        root = fetch(ids[i:i + 20])
        for e in root.findall("a:entry", NS):
            aid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.rsplit("/abs/", 1)[1])
            title = e.find("a:title", NS).text
            authors = [a.find("a:name", NS).text for a in e.findall("a:author", NS)]
            year = e.find("a:published", NS).text[:4]
            cat = e.find("arxiv:primary_category", NS).attrib["term"]
            entries[aid] = (title, authors, year, cat)
        time.sleep(3)
    out, missing = [], []
    for key, aid in KEYS.items():
        if aid not in entries:
            missing.append((key, aid))
            continue
        title, authors, year, cat = entries[aid]
        auth = " and ".join(ascii_name(a) for a in authors[:12]) + (" and others" if len(authors) > 12 else "")
        out.append(f"@article{{{key},\n  title = {{{{{tex_escape(title)}}}}},\n  author = {{{auth}}},\n  year = {{{year}}},\n"
                   f"  journal = {{arXiv preprint arXiv:{aid}}},\n  eprint = {{{aid}}},\n  archivePrefix = {{arXiv}},\n"
                   f"  primaryClass = {{{cat}}},\n}}\n")
    open("paper/refs.bib", "w", encoding="utf-8").write("\n".join(out))
    print(len(out), "entries written;", "missing:", missing)


if __name__ == "__main__":
    main()
