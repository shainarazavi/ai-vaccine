# Just as Humans Need Vaccines, So Do Models: Model Immunization to Combat Falsehoods

[![arXiv](https://img.shields.io/badge/arXiv-2505.17870-b31b1b.svg?logo=arxiv&logoColor=white)](https://www.arxiv.org/abs/2505.17870)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://shainarazavi.github.io/ai-vaccine/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official implementation of the paper **"Just as Humans Need Vaccines, So Do Models: Model Immunization to Combat Falsehoods"**.

> **Abstract:** Large language models (LLMs) reproduce misinformation by learning the linguistic patterns that make falsehoods persuasive, such as hedging, false presuppositions, and citation fabrication, rather than merely memorizing false facts. We propose model immunization: supervised fine-tuning on curated (false claim, correction) pairs injected as small "vaccine doses" (5–10% of tokens) alongside truthful data. Unlike post-hoc filtering or preference-based alignment, immunization provides direct negative supervision on labeled falsehoods. Across four open-weight model families, immunization improves TruthfulQA accuracy by 12 points and misinformation rejection by 30 points with negligible capability loss. We outline design requirements, which includes, dosage, labeling, quarantine, diversity and call for standardized vaccine corpora and benchmarks that test generalization, making immunization a routine component of responsible LLM development.

![Teaser](./images/main.png)

## BibTeX

```bibtex
@article{ModelImmunization2026,
  title={Just as Humans Need Vaccines, So Do Models: Model Immunization to Combat Falsehoods},
  author={Shaina Raza, Rizwan Qureshi, Azib Farooq, Marcelo Lotif, Aman Chadha, Deval Pandya, Christos Emmanouilidis},
  journal={WCCI},
  year={2026},
  url={https://www.arxiv.org/abs/2505.17870}
}
```
