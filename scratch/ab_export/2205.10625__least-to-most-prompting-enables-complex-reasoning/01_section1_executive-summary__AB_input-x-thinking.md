# Section 1 — Executive Summary — A/B

**2205.10625 — Least-to-Most Prompting Enables Complex Reasoning in Large Language Models**


_input chars: PyMuPDF=202,852 · GLM-OCR=240,639_



## PyMuPDF input · THINK-HIGH

## 1. Executive Summary

This paper introduces **least-to-most prompting**, a two-stage strategy where a language model first decomposes a complex problem into easier subproblems (e.g., breaking a long SCAN command like “look opposite right thrice after walk” into short commands “look opposite right”, “look opposite right thrice”, “walk”) and then solves them sequentially, using the answers to earlier subproblems to facilitate later ones. Evaluated on symbolic manipulation (last-letter concatenation with list lengths up to 12), compositional generalization (the SCAN benchmark, including the length split), and math reasoning (GSM8K, DROP) with the `code-davinci-002` model, least-to-most prompting achieves 99.7% accuracy on SCAN’s length split using only 14 exemplars—compared to 16.2% for chain-of-thought prompting—and reaches 74.0% on 12-letter concatenation (vs. 31.8%). On GSM8K, it lifts accuracy on problems requiring ≥5 reasoning steps from 39.07% to 45.23%, establishing that easy-to-hard generalization works when the required decomposition is straightforward.


## GLM-OCR input · THINK-HIGH

## 1. Executive Summary

This paper proposes **least-to-most prompting**, a two-stage prompting strategy that teaches large language models to break down complex problems into simpler subproblems (e.g., decomposing a long SCAN command like “look opposite right thrice after walk” into a sequence of short commands, or splitting “How many times can she slide?” into “How long does each trip take?”) and then solve them sequentially, using the answer to each subproblem as a building block for the next. On the SCAN compositional generalization benchmark, least-to-most prompting with GPT-3 code-davinci-002 achieves 99.7% accuracy under the length split using only 14 exemplars—far exceeding specialized neural-symbolic models trained on the full 15,000+ example dataset—compared to 16.2% for chain-of-thought prompting; on symbolic manipulation, it reaches 74.0% for 12-letter lists versus chain-of-thought’s 31.8%, and on math reasoning it lifts accuracy for problems requiring at least 5 steps from 39.07% to 45.23%. The approach establishes that decomposing problems into simpler pieces enables large language models to generalize to harder-than-exemplar instances, but its gains are most dramatic when the task’s decomposition structure is easily learned from a few examples, as in SCAN and last-letter concatenation, while in math word problems the benefit concentrates on multi-step reasoning.


## PyMuPDF input · THINK-NONE

## 1. Executive Summary

This paper introduces **least-to-most prompting**, a two-stage prompting strategy that teaches large language models to solve complex problems by first decomposing them into a series of simpler subproblems and then solving those subproblems sequentially—each solution building on answers to previously solved subproblems. Evaluated on symbolic manipulation (last-letter-concatenation with list lengths 4–12), compositional generalization (the SCAN benchmark, including its challenging length split), and math reasoning (GSM8K and DROP) using the GPT-3 `code-davinci-002` model, least-to-most prompting enables generalization to problems substantially harder than the exemplars shown in the prompt. On SCAN’s length split—where neural-symbolic models in the literature are typically trained on over 15,000 examples—least-to-most prompting achieves 99.7% accuracy using only 14 exemplars, compared to 16.2% with chain-of-thought prompting, establishing that easy-to-hard generalization via problem decomposition is effective even when the base model receives no training or fine-tuning.


## GLM-OCR input · THINK-NONE

## 1. Executive Summary

This paper introduces **least-to-most prompting**, a strategy that teaches large language models to solve complex problems by first decomposing them into simpler subproblems and then solving those subproblems sequentially using the answers to previous ones. Evaluated on GPT-3’s `code-davinci-002` model across symbolic manipulation (last-letter concatenation), compositional generalization (SCAN), and math reasoning (GSM8K, DROP), least-to-most prompting achieves 99.7% accuracy on SCAN’s length split with only 14 exemplars—compared to 16.2% for chain-of-thought prompting—and lifts accuracy on GSM8K problems requiring ≥5 steps from 39.07% to 45.23%, establishing that decomposition-driven prompting enables generalization to harder problems than those seen in the exemplars only when the decomposition itself is learnable from few-shot demonstrations.