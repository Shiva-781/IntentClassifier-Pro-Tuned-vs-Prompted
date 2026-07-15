# IntentClassifier Pro: Tuned vs Prompted

## Project Overview
IntentClassifier Pro is an NLP project that classifies user intents in banking and support-related queries using a fine-tuned DistilBERT model.

The project compares:

1. Fine-tuned DistilBERT (LoRA-inspired efficient training approach)
2. Prompting-based baseline

---

## Technologies Used

- Python
- Hugging Face Transformers
- Datasets
- PyTorch
- Streamlit
- Scikit-learn
- Google Colab

---

## Dataset

Dataset: CLINC150 (Banking and Support Intents)

- Number of intents: 61
- Train samples: 15,250
- Validation samples: 3,100
- Test samples: 5,500

---

## Model

Base Model:

```text
distilbert-base-uncased
```

Fine-tuned for multi-class intent classification.

---

## Results

| Method | Accuracy | Latency |
|---------|-----------|----------|
| Prompting Baseline | ~72% | 2-5 sec |
| Fine-tuned DistilBERT | 86.06% | <1 sec |

---

## Error Analysis

| Input | Expected | Predicted |
|-------|-----------|------------|
| I forgot my password | forgot_password | pin_change |
| I want to block my card | card_block | damaged_card |

---

## Project Outcomes

- Trained an intent classifier and deployed a working demo.
- Benchmarked fine-tuned model against prompting baseline.
- Performed confusion matrix and error analysis.

---

## Demo

Streamlit App:

PASTE_YOUR_STREAMLIT_LINK_HERE

Hugging Face Model:

https://huggingface.co/Sibbu07/intent-classifier-pro

---

## Repository Structure

```text
app.py
requirements.txt
README.md
IntentClassifier_Pro.ipynb
confusion_matrix.png
```

---

## Author

Shiva
