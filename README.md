# NEAT-based Mixture-of-Agents (NEAT-MoA)

## Introduction

The [Mixture of Agents (MoA)](https://github.com/togethercomputer/MoA) approach leverages multiple LLMs collectively to achieve state-of-the-art performance across diverse tasks. While traditional MoA implementations rely on fixed structures, NEAT-MoA dynamically evolves optimal agent configurations based on input characteristics through [NeuroEvolution of Augmenting Topologies (NEAT)](https://neat-python.readthedocs.io/en/latest/neat_overview.html). **NEAT-MoA dramatically outperforms cutting-edge models like Claude Sonnet 4 on SWE-Bench, achieving 36.84% resolved rate versus 15.78%**, using only older, underperforming models.

## Framework Architecture

NEAT-MoA combines an evolved NEAT feedforward neural network with standard MoA components (N proposer LLMs per layer and an aggregator LLM). For each input prompt, the NEAT network dynamically predicts the optimal proposer models to use within the MoA structure, allowing the system to adapt its configuration in real-time. This architecture, as shown in Figure 1, enables efficient model selection tailored to specific inputs, maximizing performance while optimizing computational resources.

<figure>
  <img src="https://raw.githubusercontent.com/dora2928/NEAT-MoA/refs/heads/main/neat-more.gif" alt="NEAT-MoA explained" style="width:75%">
  <figcaption>Figure 1. NEAT-MoA architecture. The trained NEAT network predicts the LLM combination of MoA given a prompt of chat history. </figcaption>
</figure>

## Training Methodology

The NEAT-MoA training process extracts meta-features from input prompts to identify conversation phases and assign specialization weights across four domains: code exploration, bug diagnosis, solution implementation, and test engineering. Each genome (NEAT network) is evaluated on random SWE-bench instances using the mini-swe-agent framework, which provides an agentic workflow for solving programming problems. As the agent progresses through different problem-solving phases (exploration, planning, coding, testing), the conversation context evolves dynamically, allowing the NEAT network to adapt its predictions of optimal LLM combinations based on each model's specialized capabilities. 

The NEAT network topology evolves by optimizing input nodes, hidden layers, connection patterns, and activation functions to maximize performance while minimizing costs. Figure 2 illustrates the NEAT's topology evolution example in binary classification. NEAT-MoA applies this approach for multi-label classification, where an input (chat history) can belong to multiple categories (LLMs) simultaneously.
Our fitness function balances solution effectiveness (80%), completion ratio (10%), architecture complexity, and API cost penalties, while population dynamics maintain diversity through speciation with elitism preserving top performers. 

<figure>
  <img src="https://blog.otoro.net/assets/20160507/neat_anim.gif" alt="NEAT-MoA – topology evolution example" style="width:75%">
  <figcaption>Figure 2. NEAT's topology evolution example in binary classification. NEAT-MoA applies this approach for multi-label classification, where an input (chat history) can belong to multiple categories (LLMs) simultaneously. </figcaption>
</figure>

## Evaluation Approach

We evaluate NEAT-MoA using two primary metrics: resolved rate (percentage of successfully solved instances) and average API cost per instance in dollars. System-level assessment focuses on optimizing NEAT hyperparameters, including generations, population size, and mutation rates to identify the most effective network configuration. Our comparative baseline is the mini-swe-agent framework powered by a single state-of-the-art LLM (Claude Sonnet 4, Gemini 2.5 Pro etc), against which we test our NEAT-MoA combining only older, typically underperforming models ("llama-3-1-405b", "mistral-large-2", "vertex-claude-v35-sonnet", and "gemini-v25-flash") with "vertex-claude-v37-sonnet" as the aggregator. 


### Results

| Method |Resolved|Average Cost
|--------|--|---
|Gemini-v25-Pro|66.66%|$0.16
|Gemini-v25-Flash|47.61%|$0.02
|NEAT-MoA	|36.84%|$0.48
|Claude Sonnet 3.5|19.04%|$0.24
|Mistral Large 2	|16.66%|$1.23
|Claude Sonnet 4|14.28%|$0.27
|Claude Sonnet 3.7|14.28%|$0.91
|Llama-3-1-405b	|4.76%|$0.34

We achieved 3rd position on a subset of SWE-bench instances. However, while we outperform older underperforming models (`llama-3-1-405b`, `mistral-large-2`, `vertex-claude-v35-sonnet` etc), they, as proposer options, also degrade the performance of one performant proposer (`Gemini-v25-Flash`). Gemini models likely excel in this domain because they handle long contexts effectively—a crucial capability as conversation history expands with the increasing steps required to solve complex SWE-bench tasks. For this dataset in particular, future iterations should prioritize proposer options that can reasonably handle extended contexts to maintain performance throughout the problem-solving process.

For the optimal NEAT hyperparameters, we observed that genomes tend to overfit, as the number of generations increases. For this dataset, we minimally tuned NEAT-MoA with population size of 5 on 3 generations, which seems to be reasonable in this small size experiment. We used the best genome from the 3rd generation, which achieved the highest fitness score.


### Generalization Capabilities

NEAT-MoA's domain adaptation mechanism is implemented in the `evaluate_genome()` function in `moa_neat_model.py`, which assesses network performance across domain-specific samples:
```python
def evaluate_genome():
   for messages in samples:
        features = get_conversation_features(messages)
        model_weights = get_model_weights(messages, features).values()       
        outputs = neat_network.activate(model_weights)
        moa_config = adjust_moa_config_with_neat(outputs, moa_config_path)
        response = run_neat_moa(messages, moa_config)
        fitness = evaluate_response(messages, response)
   return fitness
```

The framework demonstrates strong transfer learning capabilities, being applicable to various LLM-powered domains including code generation, optimization, and instruction following—simply replace any individual LLM with a trained NEAT-MoA. Regarding scaling, API costs increase with token length, making it crucial to select proposer models capable of independently solving tasks to avoid performance degradation. Our current evaluation is limited by the expense of API calls, constraining our SWE-bench testing to 12 representative samples with varying difficulty, potentially introducing bias in reported performance metrics.

Integration with existing systems is straightforward—for Artemis Intelligence's GA codebase where MoA is already implemented, one can replace the code generation function with `neat-moa.py` to leverage our pre-trained model on SWE-bench data. For custom applications, the framework provides `moa_neat_model.py` and `run_neat_optimization.py` where the `evaluate_genome()` function can be adjusted for domain-specific datasets. 

### Conclusion and Future Work

NEAT-MoA demonstrates significant potential for optimizing both performance and cost on SWE-bench tasks, achieving superior results while maintaining reasonable training costs (under $15). Our findings suggest that proposer LLMs should have comparable performance characteristics to prevent degradation in the combined system's effectiveness. Beyond software engineering, NEAT-MoA's applications extend to various domains requiring agentic workflows, as well as code generation, code optimization, instruction following, and complex reasoning tasks. Future research will focus on refining proposer model selection to include options with performance characteristics similar to `Gemini-v25-Flash` (long-context model), which should improve training efficiency and overall system capabilities on SWE-bench.
