---
base_model: allenai/Olmo-3-1025-7B
library_name: transformers
model_name: olmo-contaminated-20260122_214250
tags:
- generated_from_trainer
- sft
- trl
licence: license
---

# Model Card for olmo-contaminated-20260122_214250

This model is a fine-tuned version of [allenai/Olmo-3-1025-7B](https://huggingface.co/allenai/Olmo-3-1025-7B).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- TRL: 0.27.0
- Transformers: 4.57.6
- Pytorch: 2.10.0
- Datasets: 4.5.0
- Tokenizers: 0.22.2

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```