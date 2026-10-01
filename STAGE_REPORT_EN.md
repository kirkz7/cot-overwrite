# Last Word Wins? How language models resolve conflicting state (progress report)

> 2026-10-01 · 22 experiments · 15 models (1.7B–14B: Qwen3 / OLMo-2 / Phi-4 / R1-Distill) · local RTX 5080
> Web versions with interactive figures: English https://claude.ai/artifact/2FfEBT2PQsUoZ8vZRU3xZp · 中文 https://claude.ai/artifact/SingYsTQiWeNEwnqjswiZL
> Chinese markdown: `STAGE_REPORT.md`; per-experiment log: `RESULTS.md`; literature: `LITERATURE.md`; roadmap: `PLAN.md`

## Summary

When a language model reads text that states several values for the same quantity, it **defaults to the value presented last**. For text written in chronological order, including the model's own reasoning, this default is usually right. When the text is not presented in chronological order, the default overrides explicit temporal markers (step numbers, “listed from most recent”), and an explicit retraction (“Wait, that's wrong.”) only partly offsets it. Qwen3-14B can use well-structured step numbers; OLMo-2-13B, at the same scale, still reads mostly by position.

**Where this stands**: evidence across two model families and 1.7B–14B supports a paper on position-based reading of state. Whether it is main-conference material depends on whether the effect persists at 32B and 70B, and whether it reproduces in natural non-chronological settings (agent memory retrieval, newest-first logs, multi-source contexts).

| Claim | Status | Key evidence |
|---|---|---|
| Order matters only when states are overwritten | Supported | All 15 models on the synthetic task; across all 27 BBH tasks, shuffling breaks 39% of answers on tasks that track updated state and 4% on the rest |
| Readers go by presentation position, not by which value is newer | Supported | With an explicit “most recent first” header, Qwen3-4B almost never picks the actually newest value (k = 2/4/8: 1/0/0%) |
| Position overrides explicit temporal cues; retractions only partly offset it | Largely supported | Qwen3-4B and OLMo-2-7B/13B reach 0–26% with newest-first order plus step numbers; Qwen3-14B is the exception (about 90%). Retractions weaken the positional advantage (OLMo-13B: 88% → 42%), but no model switches reliably to the unretracted conclusion |
| Reading its own chronological reasoning, the positional default is usually right | Supported | Re-reading its own BBH reasoning, 93–100% on most tasks; in 60% of its multiple-choice reasoning the correct option is not mentioned last, yet accuracy stays at 97.8% |
| When support is balanced the later conclusion wins; when it is not, support wins | Supported | Two conclusions stated once each: peak position effect 27–87 points depending on the reader; each extra restatement of the correct answer cancels much of the positional advantage |
| Mechanism: late-layer heads read stated values with a positional bias | Preliminary | Zeroing 32 heads cuts the real-text position effect from 42 to 10 points (32 random heads: 44); activation patching is needed to separate reading from positional bias |
| Post-training raises sensitivity to retraction markers only at sufficient scale | Two families | OLMo-2 7B: 5.6 → 9.1%; 13B: 9.6 → 29.0%; Qwen3-4B: 6.8 → 33.1% |

## Evidence

### 1 · When order matters: only when states are overwritten (Exps 1, 8, 20)

- Synthetic task (Qwen3-4B, 200 items per cell): a 16-line variable-tracking program. After the reasoning is shuffled, accuracy on the overwritten variable is 100 / 79.5 / 39.5 / 19.0 / 14.0% at k = 0/1/2/4/8 (chance 100 / 50 / 33 / 20 / 11); once-assigned variables stay ≥99.5%. The model picks the value presented last in 72 / 93.5 / 97 / 98.5% of cases at k = 1/2/4/8, and in 93–99% up to 64 overwrites.
- BBH with the model's own reasoning (answer sentence removed, line-level shuffle × 3): tracking with three / five / seven objects 100 → 51.3, 89.6 → 29.0, 82.9 → 26.1; logical deduction with five objects 89.6 → 85.3. By swaps involving the queried person, k = 1/2/3/4: 41.1 / 42.4 / 28.3 / 21.7% after shuffling.
- Prior work: LCA (2605.26795) and 2605.07307 report that shuffling does not hurt; 2605.22870 reports collapse on tracking. Whether states are overwritten explains both. 2605.22870 also reports collapse on logical deduction, which we do not observe (different setup); the paper needs to address this.

### 2 · Explicit cues lose to position (Exps 18, 19, 2) — core

**Exp 18 (accuracy % in the conflict case, k = 2 / 4 / 8)**

| Condition | Qwen3-4B | Qwen3-14B | OLMo-2-13B | OLMo-2-7B |
|---|---|---|---|---|
| Shuffled, no cue | 5.5 / 1.8 / 0.6 | 24.2 / 3.6 / 1.2 | 7.0 / 5.5 / 0.0 | 0.8 / 0.6 / 1.7 |
| Shuffled + step number on every line | 25.0 / 17.6 / 8.7 | 64.1 / 48.5 / 30.6 | 46.1 / 43.6 / 43.9 | 24.2 / 21.8 / 11.0 |
| Newest-first + “most recent first” header | 1.0 / 0 / 0 | 69.0 / 41.0 / 12.0 | 19.5 / 7.0 / 6.5 | 0 / 0 / 0 |
| Newest-first + step number on every line | 14.0 / 3.0 / 5.0 | 90.5 / 87.5 / 88.5 | 26.0 / 11.5 / 11.5 | 1.0 / 0 / 0 |

Removing the word “final” from the answer stem does not change the results. Qwen3-14B uses step numbers in newest-first lists; OLMo-2-13B at the same scale does not.

**Exp 19 (symmetric control: two counterfeit wrong conclusions, A earlier and B later, correct answer absent; 10% of the derivation kept; % choosing A / B)**

| | Qwen3-4B | Qwen3-14B | OLMo-2-13B |
|---|---|---|---|
| No marker | 11.4 / 71.6 | 20.0 / 59.5 | 7.0 / 88.0 |
| B followed by “Wait, that's wrong.” | 19.4 / 56.5 | 27.5 / 45.5 | 40.0 / 41.5 |

A rational reader should pick A when B is retracted.

**Exp 2 (synthetic)**: when a wrong value comes last and is followed by “Hmm, that's wrong.”, most models reach only 0.5–10% in the conflict case; the best, Qwen3-14B, reaches 47.5%.

### 3 · Real reasoning text: support first, position breaks ties (Exps 13, 15, 16 and others)

In R1 traces from OpenR1-Math, the last paragraph stating the correct answer is copied with the answer replaced by a nearby wrong number X; both conclusions appear once, in either order, and derivation paragraphs are progressively removed.

**Position effect (correct conclusion last − wrong conclusion last, points) by share of derivation kept, 0 / 10 / 30 / 60 / 100%:**

| Reader | 0% | 10% | 30% | 60% | 100% |
|---|---|---|---|---|---|
| Qwen3-4B | 21.7 | 42.0 | 32.2 | 19.0 | 10.0 |
| Qwen3-8B | 19.5 | 44.0 | 36.5 | 24.5 | 17.0 |
| Qwen3-14B | 14.0 | 27.0 | 24.0 | 14.3 | 8.0 |
| R1-Distill-Qwen-7B | 7.0 | 51.5 | 61.0 | 52.0 | 42.5 |
| OLMo-2-7B-Instruct | 80.0 | 85.5 | 82.5 | 84.5 | 72.0 |
| OLMo-2-13B-Instruct | 63.5 | 76.0 | 76.0 | 76.0 | 66.0 |
| Phi-4-mini | 86.0 | 86.5 | 86.5 | 84.0 | 85.5 |

**Restatements vs position (Exp 15, % choosing the wrong conclusion X, r = 0 / 1 / 2 / 4 extra restatements)**

| Condition | Qwen3-4B | Qwen3-14B |
|---|---|---|
| Correct conclusion restated, wrong conclusion last | 37.3 / 19.7 / 13.3 / 10.7 | 30.0 / 13.0 / 7.0 / 4.5 |
| Wrong conclusion restated, correct conclusion last | 6.7 / 28.0 / 37.7 / 51.0 | 7.0 / 34.0 / 43.0 / 57.0 |

**In complete natural traces a single edit does not move the answer**: shuffling paragraphs (99% correct in the conflict case), inserting a retraction sentence (X chosen 0–0.3%), a counterfeit conclusion paragraph (0–0.8%), swapping two natural conclusions in R1 traces that end wrong (final answer kept 88–100%).

**Exp 17 (answer extractor)**: when a solution ends with a wrong answer X, Qwen3-4B and 14B still report the correct answer 69% and 67.5% of the time, so the extractor is not faithful to the solution. Answer recognition is a confound; a symmetric design is still needed.

### 4 · Scope: the model's own reasoning versus other text it reads

The failures above occur when text is not presented in chronological order, or when two conclusions have equal support. A model's own reasoning is almost always chronological, so unedited reasoning was tested separately:

| Qwen3-4B re-reading its own reasoning (BBH, answer sentence removed only) | Value |
|---|---|
| Accuracy re-reading in the original order (most tasks) | 93–100% |
| Multiple-choice items whose reasoning discusses ≥2 options | 302 |
| …of which the correct option is not the last one mentioned | 60.3% |
| Re-read accuracy: correct option mentioned last / not last | 100% / 97.8% |
| R1 traces ending wrong that first stated the correct answer | 7.6% |

**Conclusion**: mismatches between position and answer are common in a model's own reasoning, but explicit judgments (“(B) is correct”) usually resolve them, at a cost of about 2 points. The claim of this work is therefore not that models misread their own reasoning. It is that **when a model reads state updates that are not in chronological order and lack explicit judgments, it takes the value by position.** Such text is common in deployed systems: agent memory retrieval returns old and new facts by relevance; git logs, email threads and changelogs list the newest entry first; tool outputs, conversation history and other models' reasoning are concatenated (as in CoT monitoring); and long reasoning revises the same quantity many times.

### 5 · Across tasks: all 27 BBH tasks (Exp 22)

Qwen3-4B's own reasoning, 3 line-level shuffles per item; only items re-read correctly in the original order count. Task categories were written into the code before the experiment ran.

| Category assigned in advance | Mean broken | Tasks (% broken) |
|---|---|---|
| Tracks updated state | **39%** | Tracking 7 / 5 / 3: 72.6 / 70.5 / 56.5; multi-step arithmetic 42.0; navigate 20.8; object counting 5.6; boolean expressions 5.3 |
| Partly | 10% | Word sorting 18.3; date understanding 7.0; colored objects 6.3; penguins in a table 6.2 |
| No | **4%** | Deduction 5 / 7 / 3: 10.3 / 8.3 / 1.4; web of lies 9.4; movie recommendation 6.5; ruin names 5.4; causal judgement 5.1; temporal sequences 3.7; hyperbaton 3.4; disambiguation QA 3.1; snarks 1.4; formal fallacies 0.8; translation error detection 0.6; geometric shapes 0; sports understanding 0 |

Object counting and boolean expressions were labeled state-tracking in advance, but the model's reasoning lists items and evaluates sub-expressions one at a time without overwriting a quantity, so they rarely break. What matters is whether the reasoning text actually overwrites state. Dyck languages had a single usable item and is excluded.

### 6 · Mechanism (Exps 7 and 14, Qwen3-4B)

| Heads zeroed | Synthetic shuffled accuracy | Synthetic shuffled picks last | Synthetic ordered accuracy | Real-text position effect (10% derivation) |
|---|---|---|---|---|
| None | 19.0 | 97.0 | 99.5 | 41.7 |
| Top 16 | 24.5 | 80.0 | 97.0 | — |
| Top 32 | 20.5 | 68.5 | 84.0 | 10.3 |
| 32 random | 18.0 | 97.0 | 100 | 44.3 |

The top heads sit in layers 18–29 (L29H11 puts 81% of its value attention on the last line). On real text the effect drops mainly because accuracy with the correct conclusion last falls from 91.7% to 56.0%: these heads are how the model reads stated values and conclusions, and the reading itself carries the positional bias.

### 7 · Scale and training

| Metric | Qwen3-4B | Qwen3-14B | OLMo-2-7B-Inst | OLMo-2-13B-Inst |
|---|---|---|---|---|
| Synthetic shuffled, k=8: picks last value % | 98.5 | 97.0 | 95.5 | 95.0 |
| Synthetic conflict, both markers: accuracy % | 33.1 | 47.5 | 9.1 | 29.0 |
| Newest-first + step numbers, k=8 % | 5.0 | 88.5 | 0 | 11.5 |
| Real, symmetric: later conclusion chosen despite retraction % | 56.5 | 45.5 | — | 41.5 |
| Real: position effect (10% derivation) | 42.0 | 27.0 | 85.5 | 76.0 |
| Real: position effect (100% derivation) | 10.0 | 8.0 | 72.0 | 66.0 |

Retraction-marker sensitivity (Exp 2 conflict accuracy, both markers): Qwen3-1.7B 3.3; Qwen3-4B-Base 6.8 → Qwen3-4B 33.1; Qwen3-8B 32.3; Qwen3-14B 47.5; OLMo-2-7B 5.6 / SFT 6.3 / DPO 9.3 / Instruct 9.1; OLMo-2-13B 9.6 → Instruct 29.0; Qwen2.5-Math-7B (base) 32.3 → R1-Distill-Qwen-7B 17.7; Phi-4-mini 5.6.

On real text Qwen3 shows smaller position effects than non-Qwen readers, possibly because Qwen3 has seen these problems; with that removed by the symmetric control, the effect is still 60 points for 4B and 40 points for 14B.

## Risks and unknowns

1. **The effect weakens with scale (biggest risk)**: Qwen3-14B handles well-structured step numbers and OLMo-2-13B does not, which points to training data or model family rather than scale alone. If 32B and 70B models handle the hard conditions, the claim narrows to small and mid-sized readers.
2. **Reproduction in natural settings (key)**: all conflicts so far are constructed under control. Reading its own reasoning costs the model little; the effect has to be shown in natural non-chronological settings such as agent memory retrieval and newest-first logs.
3. Models of 7B and above run in 4-bit; key results need bf16 replication.
4. Only reading is measured; whether retracted values are reused during generation is untested.
5. Logical deduction on BBH disagrees with 2605.22870.
6. Test of Time, 2510.22752 and LongMemEval need a close read and a clear distinction.

## Next steps and decisions

**1. Natural non-chronological settings (feasible on the RTX 5080)**
- **Agent memory retrieval**: on a long-term memory benchmark with knowledge updates, compare retrieved results ordered by relevance (with timestamps) against the same results ordered by time. If relevance order is clearly worse, re-sorting by time is a free fix.
- **Newest-first logs**: build “what is the current value” questions from real commit histories and changelogs, presented oldest-first and newest-first.
- **The model's own long reasoning**: measure how often thinking-mode traces revise the same quantity, and whether later steps reuse retracted values.

**2. Rent a GPU to test scale**: one 80GB card (A800 / H800 / H100), about 30–50 hours. First run Exps 18, 19, 13 and 2 on 32B models (Qwen3-32B, OLMo-2-32B), about 10–15 hours, then decide on 70B. Also activation patching, bf16 replication of key results, and the generation-side experiments.

**Decisions for the advisor:**
1. **Framing**: the paper is about language models reading state updates by presentation position, with controlled experiments establishing the phenomenon and mechanism, and memory retrieval and logs showing the impact.
2. **Compute**: whether to rent a GPU, the budget, and whether the lab has cloud credits.
3. **Which natural setting first**: agent memory retrieval, newest-first logs, CoT monitoring, or the model's own long reasoning.
4. **Target venue and timeline**.

## Experiment index

| # | Question | Status |
|---|---|---|
| 1 | Shuffle × overwrites (synthetic) | Done |
| 2 | Position vs retraction markers (synthetic) | Done |
| 3 | Cross-model replication | Done |
| 4 | Paragraph shuffle of real traces | Done |
| 5 | Natural-language synthetic task | Done |
| 6 | Early answering + shuffle | Done |
| 7 | Recency heads + ablation | Done |
| 8 | Long overwrite chains (k up to 64) | Done |
| 9 | Post-training lineage + scale | Done |
| 10 | Read/write asymmetry (generation side) | Planned |
| 11 | Inserted retraction sentence | Done |
| 12 | Counterfeit conclusion paragraph | Done |
| 13 | Competing conclusions, dose-response | Done |
| 14 | Ablating recency heads (real text) | Done |
| 15 | Restatement vs position | Done |
| 16 | More readers for Exp 13 | Done |
| 17 | Answer extractor + mitigation prompt | Done |
| 18 | Explicit order cues vs position | Done |
| 19 | Real-text retraction + symmetric control | Done |
| 20 | BBH tracking vs deduction | Done |
| 21 | Natural competing conclusions | Done |
| 22 | All 27 BBH tasks | Done |
