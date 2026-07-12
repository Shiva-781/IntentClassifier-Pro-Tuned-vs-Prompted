
import streamlit as st
import transformers

st.write("Transformers version:", transformers.__version__)

try:
    from transformers import AutoTokenizer
    st.success("AutoTokenizer imported successfully")
except Exception as e:
    st.error(f"AutoTokenizer Error: {e}")

try:
    from transformers import AutoModelForSequenceClassification
    st.success("AutoModel imported successfully")
except Exception as e:
    st.error(f"AutoModel Error: {e}")

st.stop()
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "Sibbu07/intent-classifier-pro"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    return tokenizer, model

tokenizer, model = load_model()

st.title("IntentClassifier Pro")
st.write("Predict the intent of a sentence using a fine-tuned DistilBERT model.")

text = st.text_area(
    "Enter your sentence",
    placeholder="Example: I want to transfer money to my savings account"
)

if st.button("Predict Intent"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=64
        )

        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()

        labels = model.config.id2label

        if isinstance(list(labels.keys())[0], str):
            intent = labels[str(pred)]
        else:
            intent = labels[pred]

        st.success(f"Predicted Intent: {intent}")
