# Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters

This paper shows how to optimally allocate a fixed test-time compute budget for large language models by adaptively choosing between search and iterative revision strategies based on prompt difficulty. The resulting compute-optimal strategy improves efficiency by over 4× compared to best-of-N baselines and enables a smaller model to outperform a ~14× larger model on easy-to-medium math problems, demonstrating that strategic inference-time computation can substitute for pretraining scale.
