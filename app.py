import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🤖 HR Policy Assistant")
st.write(
    "Ask questions about company HR policies. "
    "Answers are generated only from the provided HR Policy."
)


# --------------------------------------------------
# Load Models
# --------------------------------------------------

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    llm_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name
    )

    return embedding_model, tokenizer, llm_model


embedding_model, tokenizer, llm_model = load_models()


# --------------------------------------------------
# Read HR Policy PDF
# --------------------------------------------------

@st.cache_resource
def load_policy():

    reader = PdfReader("HR_Policy.pdf")

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted + "\n"

    return text


# --------------------------------------------------
# Section-based Chunking
# --------------------------------------------------

def create_chunks(text):

    sections = text.split("\n")

    chunks = []
    current_chunk = ""

    for line in sections:

        line = line.strip()

        if not line:
            continue

        if line[0].isdigit() and "." in line[:3]:

            if current_chunk:
                chunks.append(current_chunk.strip())

            current_chunk = line

        else:
            current_chunk += " " + line

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# --------------------------------------------------
# Create Vector Database
# --------------------------------------------------

@st.cache_resource
def create_vector_store():

    text = load_policy()

    chunks = create_chunks(text)

    embeddings = embedding_model.encode(chunks)

    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="hr_policies"
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist()
    )

    return collection


collection = create_vector_store()


# --------------------------------------------------
# Generate Answer
# --------------------------------------------------
def generate_answer(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=60,
        num_beams=4,
        do_sample=False,
        early_stopping=True
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()
# --------------------------------------------------
# RAG Function
# --------------------------------------------------

def ask_rag(question):

    question_embedding = embedding_model.encode(
        [question]
    )

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=1
    )

    retrieved_chunks = results["documents"][0]
    distance = results["distances"][0][0]

    # Hallucination protection
    threshold = 1.0

    if distance > threshold:

        return (
            "The information is not available "
            "in the provided HR policy.",
            []
        )

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
Answer the question using only the information in the context.

Context:
{context}

Question:
{question}

Give a short complete answer.
Do not guess.
Do not add information that is not in the context.

Answer:
"""

    answer = generate_answer(prompt)

    return answer, retrieved_chunks

    answer = generate_answer(prompt)

    return answer, retrieved_chunks


# --------------------------------------------------
# User Interface
# --------------------------------------------------

question = st.text_input(
    "Ask your HR policy question:",
    placeholder="Example: How many casual leaves can an employee take?"
)


if st.button("Ask 🤖"):

    if question.strip():

        with st.spinner("Searching HR policy..."):

            answer, sources = ask_rag(question)

        st.subheader("🤖 Answer")

        st.write(answer)

        st.divider()

        st.subheader("📚 Retrieved Source")

        if sources:

            for i, source in enumerate(sources):

                st.info(
                    f"Source {i + 1}\n\n{source}"
                )

        else:

            st.warning(
                "No relevant source found in the HR policy."
            )

    else:

        st.warning("Please enter a question.")
