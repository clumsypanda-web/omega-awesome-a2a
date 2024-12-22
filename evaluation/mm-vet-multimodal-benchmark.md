Add: MM-Vet Multimodal Benchmark Evaluation Framework
# MM-Vet: Integrated Capabilities Benchmark for Large Multimodal Models

**Link:** https://github.com/yuweihao/MM-Vet

**Paper:** https://arxiv.org/abs/2308.02490

**Description:**
MM-Vet introduces a sophisticated evaluation framework that examines large multimodal models (LMMs) through the lens of integrated capabilities. Unlike traditional benchmarks, it systematically evaluates how models combine 6 core vision-language capabilities across 16 different integration scenarios. The framework includes a novel LLM-based evaluator that enables consistent scoring across diverse question types and answer formats, revealing that even advanced models like GPT-4V achieve only 68% performance, highlighting significant room for improvement in multimodal reasoning.

**Key Features:**
- Systematic evaluation of 6 core vision-language capabilities
- Novel LLM-based evaluation metric
- Online evaluation platform
- Comprehensive test cases for integrated capabilities

**Technical Implementation:**
```python
# Example evaluation code using MM-Vet
from mmvet import MMVetEvaluator

evaluator = MMVetEvaluator()
score = evaluator.evaluate(
    model_output="Model response",
    ground_truth="Reference answer",
    capability_type="OCR+Knowledge"
)
