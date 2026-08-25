# References
Below are the references used in this notebook:
- https://www.kaggle.com/whitepaper-foundational-llm-and-text-generation
# Introduction

LLM trained in a variety of tasks on **large amounts of data** perform very well out of the box they can also be adapted to solve specific tasks where performance out of the box is not at the level desired. This process is known as **fine-tuning**. Fine-tuning is especially interesting as it requires less data and computational resources to achieve a considerable improvement in the target task. LLM can be further refine to improve its performance in the desired task by **prompt engineering** which consists in composing the prompt and the parameters of an LLM to get the desired performance. 

### Prompt engineering
#prompt_engineering
Examples of prompt engineering include providing:
- clear instructions to the LLM
- giving examples
- using keywords
- formatting
- provide additional background
There are different approaches when using prompt engineering, some examples are:
- **Few-shot prompting:** You provide the LLM with a description and a set of examples (three to five) that will help guide the LLM's response.
- **Zero-shot prompting** You provide the LLM with a prompt with instructions. The LLM relies heavily on its existing knowledge to output the correct response.
- **Chain-of-thought prompting** this technique aims to improve performance on complex reasoning tasks. You provide a prompt that demonstrates how to resolve a similar problem using step-by-step reasoning.

### Sampling technique
#sampling_technique
It consist in define how the model choses the output token(s), controlling the quality, creativity and diversity of the LLM's output. Some of these techniques are:
- **Greedy search:** selects the token with the highest probability at each step.
- **Random sampling:** The next token is sampled using a probability distribution where its probability of being sampled is proportional to is predicted probability.
- **Temperature sampling:** Adjust the probability distribution by a "temperature" parameter. Higher temperature promotes diversity, lower temperature favor high-probability tokens.
- **Top-K sampling:** Randomly sample from the top K most probable tokens.
- **Top-P sampling:** Samples from a dynamic (varies) subset of tokens whose cumulative probability add up to P.
- **Best-of-N sampling:** Generates N separate responses and selects the one deemed best according to a predetermined metric. ==Particularly useful in situations where logic and reasoning are key==

```markdown
"By combining prompt engineering with sampling techniques and correctly calibrated hyperparametes, you can gratly influence the LLM's response, making it more relevant to your specific task."
```

## Task-based Evaluation
For building a tailored evaluation framework, application builders need to provide their own evaluation data, development context, and a definition of "good" performance.
- **Evaluation data:** Dedicated evaluation dataset that mirrors the expected production traffic as closely as possible.
- **Development context:** The evaluation should extend beyond the model's output to analyze the entire system, including components like *data augmentation* ( #RAG), and *agentic workflows*
- **Definition of "good":** stablish dataset criteria that reflect desired business outcomes or even rubrics that capture the core elements of the desired outputs.
## Accelerating inference
Some optimizations techniques to reduce the cost and/or speed the model response can have an impact on the model's output. For this reason, the methods used to accelerate inference will be split into two: output-approximating and output-preserving.

### Output-approximation methods
#### Quantization
it consists in reduce the weights numerical representation precision, for example moving from a 32 bit representation to a 8 or even 4 bit representation of the same number (weight). This method reduces memory load, communication overhead, speed up inference by enabling faster arithmetic operations (TPU/GPU). This approach can induce a mild to non-existent degradation in the outputs' quality.

#### Distillation
Set of techniques that target to improve a smaller model (the student) using a larger model (the teacher). One of these approaches consist of using the large model to generate synthetic data to train the smaller model. The increase in volume data will help the student model to improve.

### Output-preserving methods
these methods are guaranteed to be quality neutral, they cause no changes to the model output.

#### Flash Attention
In the context of self-attention (quadratic operation on the input length), optimizing this step can bring significant latency and cost wins. Flash attention introduced by Tri Dao et al optimized the attention algorithm by making it IO aware. They demonstrated and improvement of 2-4X latency.

#### Prefix Caching
One of the most compute intensive (slow) operations in LLM is the calculation of the attention key and value scores (KV). By keeping the KV cache we avoid recalculating attention scores.