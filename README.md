# RAG Pipeline Evaluation using RAGAS

## Overview
This project implements and evaluates a **RAG pipeline** using the **RAGAS (Retrieval-Augmented Generation Assessment)** framework. The system performs component-level evaluation of both Retriever and Generator using structured metrics to assess retrieval quality, grounding, and semantic correctness.

## What is RAGAS?
**RAGAS** stands for **Retrieval-Augmented Generation Assessment**. It is a sophisticated Python framework designed to automatically evaluate the quality of RAG systems using:
- Reference-free evaluation methods
- Multiple complementary metrics
- LLM-based assessment
- Contextual understanding
  
## RAGAS Architecture

```
Raw Generation Output
         ↓
    ┌────────────────────────────────────────┐
    │   RAGAS Evaluation Framework           │
    ├────────────────────────────────────────┤
    │ Input Components:                       │
    │ • Question/Query                        │
    │ • Generated Answer                      │
    │ • Retrieved Contexts                    │
    │ • Ground Truth (reference)              │
    └────────────────┬───────────────────────┘
                     ↓
         ┌──────────────────────────┐
         │  Metric Calculations:    │
         ├──────────────────────────┤
         │ 1. Faithfulness          │
         │ 2. Context Recall        │
         │ 3. Context Precision     │
         │ 4. Answer Correctness    │
         └────────────┬─────────────┘
                      ↓
         ┌──────────────────────────┐
         │  Aggregated Scores       │
         │  (0-1 for each metric)   │
         └────────────┬─────────────┘
                      ↓
         RAG Quality Assessment Report
```


## Two Core Evaluation Approaches in RAGAS

RAGAS evaluates RAG systems using two complementary and fundamentally different approaches:

### **Approach 1: LLM-Based Evaluation (Reasoning-Based Scoring)**

**Definition**: This approach uses an evaluator LLM to judge quality through semantic reasoning.

**How It Works**:
- The evaluator LLM is given question, answer, and/or contexts
- It reasons over the inputs using natural language understanding
- It outputs a structured score based on semantic judgment

**Used For**:
- `context_precision` - Relevance of retrieved chunks
- `context_recall` - Completeness of retrieval
- `faithfulness` - Answer grounding in context

---

### **Approach 2: Embedding-Based Evaluation (Vector Similarity)**

**Definition**: This approach uses embedding vectors to measure semantic similarity without LLM reasoning.

**How It Works**:
- Text is converted to embedding vectors (numeric representations)
- Cosine similarity is computed between vectors
- Produces deterministic score between 0 and 1

**Process**:
```
Text 1 ──→ [Embedding Model] ──→ Vector A
                                    ↓
                            [Cosine Similarity]
                                    ↑
Text 2 ──→ [Embedding Model] ──→ Vector B
```

**Used For**:
- `answer_correctness` - Semantic similarity to ground truth

---

## RAGAS Metrics

### **1️⃣ Context Precision**

**What It Measures** - 
How many retrieved chunks are actually relevant to the question (proportion of relevant chunks).

**Internal Process**:
```
Input:
  - Question: "What are heart-healthy fats?"
  - Retrieved Contexts: [chunk1, chunk2, chunk3, chunk4]
Process:
  For each context:
    LLM evaluates: "Is this context relevant to the question?"
    Output: Binary (relevant/not relevant)
  
Calculation:
  Context Precision = (# of Relevant Contexts) / (Total # of Contexts)

```

---

### **2️⃣ Context Recall** - 

**What It Measures**
Did the retriever retrieve all necessary information to answer the question? (Requires ground truth)

**Internal Process**:
```
Input:
  - Question: "What are all benefits of monounsaturated fats?"
  - Ground Truth: "Lower cholesterol, reduce stroke risk, support weight loss"
  - Retrieved Contexts: [chunk1, chunk2, chunk3]

Process:
  LLM evaluates: "Are all the ground truth facts present in contexts?"
  Checks which ground truth information is covered
  
Calculation:
  Context Recall = (# of Ground Truth Facts Covered) / (Total Ground Truth Facts)

```

---

### **3️⃣ Faithfulness** -

**What It Measures**
Is the generated answer grounded in the retrieved context? (Detects hallucinations)

**Internal Process**:
```
Input:
  - Answer: "Monounsaturated fats lower cholesterol and improve memory significantly."
  - Retrieved Contexts: "Monounsaturated fats help lower LDL cholesterol..."

Process:
  1. Break answer into factual claims:
     Claim 1: "Monounsaturated fats lower cholesterol"
     Claim 2: "Monounsaturated fats improve memory significantly"
  
  2. For each claim, LLM evaluates:
     "Is this claim supported by the context?"
  
  3. Count:
     Claim 1: ✓ SUPPORTED by context
     Claim 2: ✗ NOT SUPPORTED by context
  
Calculation:
  Faithfulness = (# of Supported Claims) / (Total # of Claims)

```

---

### **4️⃣ Answer Correctness** -

**What It Measures**
How semantically similar is the generated answer to the ground truth? (Requires ground truth).

**Internal Process**:
```
Input:
  - Ground Truth: "Monounsaturated fats lower LDL and raise HDL cholesterol"
  - Generated Answer: "Monounsaturated fats reduce bad cholesterol and increase good cholesterol"

Process:
  1. Convert both to embeddings:
     GT_embedding = embed(ground_truth)
     Gen_embedding = embed(generated_answer)
  
  2. Compute cosine similarity:
     similarity = (GT_embedding · Gen_embedding) / (||GT_embedding|| × ||Gen_embedding||)
  
  3. Score is cosine similarity value (0-1)

Example:
  Cosine Similarity = 0.92
```
