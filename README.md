# Just as Humans Need Vaccines, So Do Models: Model Immunization to Combat Falsehoods

[![arXiv](https://img.shields.io/badge/arXiv-2505.17870-b31b1b.svg?logo=arxiv&logoColor=white)](https://www.arxiv.org/abs/2505.17870)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://shainarazavi.github.io/ai-vaccine/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official implementation of the paper **"Just as Humans Need Vaccines, So Do Models: Model Immunization to Combat Falsehoods"**, and under review at **ACL 2026**.

> **Abstract:** _[Large language models reproduce misinformation not merely because they memorize false facts, but because they learn the \textit{linguistic patterns} through which falsehoods are expressed---hedged assertions, false presuppositions, fabricated citations, and rhetorical structures that lend credibility to baseless claims. We argue that improving model factuality requires moving beyond post-hoc filtering and preference-based alignment toward \textit{direct negative supervision} on explicitly labeled falsehoods. Drawing an analogy to biological immunization, we propose that models should be fine-tuned on curated corpora of fact-checked false claims paired with corrections, treating these as supervised 'vaccine doses' that teach models to recognize and reject misleading linguistic patterns. We present a framework for constructing such corpora, distinguish this approach from adversarial training and RLHF, and outline a research agenda for the NLP community---including benchmarks for misinformation robustness, standardized multilingual vaccine corpora, and evaluation protocols that test generalization to unseen falsehood types. We call on the community to develop the linguistic resources and evaluation infrastructure needed to make immunization a standard component of responsible language model development.]_

![Teaser](./images/main.png)

## BibTeX

```bibtex
@article{ModelImmunization2026,
  title={Just as Humans Need Vaccines, So Do Models: Model Immunization to Combat Falsehoods},
  author={Shaina Raza, Rizwan Qureshi, Marcelo Lotif, Azib Farooq, Aman Chadha, Deval Pandya, Christos Emmanouilidis},
  journal={ACL},
  year={2026},
  url={https://www.arxiv.org/abs/2505.17870}
}
```
