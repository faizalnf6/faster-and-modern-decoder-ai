# River Well AI: Technical Specification & Development Roadmap

## 1. Core Philosophy & Design Goals

| Field | Detail |
|---|---|
| Concept Name | River Well |
| Primary Objective | Create a language model balancing fluent generation ("River") with deep contextual retention ("Well") |
| User Experience Promise | Feels both fluent (smooth surface-level coherence) and grounded (accurate long-term memory) |

| Component | Metaphor | Technical Goal |
|---|---|---|
| RIVER | Continuous Flow | Generate coherent, unbroken reasoning streams rather than disjointed token predictions |
| WELL | Depth & Groundwater | Retrieve relevant meaning from deep context (earlier turns/documents) without degradation as the window fills |

## 2. Base Architecture (The Backbone)

Derived from the "SolsticePulse" reference implementation (GPT-2-style decoder-only Transformer).

### 2.1 Structural Components

| Component | Description |
|---|---|
| Input Layer | Token embedding + Learned position embedding (summed) |
| Decoder Blocks | Stack of pre-normalization blocks: `x = x + Attention(Norm(x))`, `x = x + MLP(Norm(x))` |
| Attention Mechanism | Multi-head causal self-attention (standard scaled dot-product) |
| Feed-Forward Network (FFN) | GELU-activated with 4x hidden-size expansion |
| Output Head | Weight tying between input embedding and output LM head |

### 2.2 Key Deviation: Normalization Strategy

| Aspect | Detail |
|---|---|
| Mechanism | RMSNorm (Root Mean Square Layer Normalization) replaces standard LayerNorm at all pre-norm points and the final output norm |
| Rescaling | By root-mean-square magnitude |
| Gain | Learned per-channel gain applied |
| Mean Subtraction | None (no re-centering) |
| Bias Term | None |
| Benefit | Computational efficiency and numerical stability at scale, crucial for supporting extended memory layers (Section 4) |

## 3. Advanced Architectural Integrations

Inspired by Cohere Command A/A+ lineage, adapted for the RMSNorm backbone.

### 3.1 Attention Optimization

| Feature | Description | Purpose |
|---|---|---|
| Interleaved Local/Global Attention | Alternates sliding-window attention layers (cheap, local) with occasional full-context attention layers (global) | Makes long-document processing tractable while maintaining global awareness |
| Grouped-Query Attention (GQA) | Shares key/value heads across groups of query heads | Reduces memory bandwidth during generation, enabling fast inference even with large context windows |

### 3.2 Activation & Scaling

| Feature | Description | Purpose |
|---|---|---|
| Gated FFN (SwiGLU-style) | Replaces plain GELU | Modest quality gains at similar compute costs |
| Mixture-of-Experts (MoE) [Future Path] | Activates only a subset of expert sub-networks per token | Allows total capacity to grow without proportional increases in per-token compute; reserved for larger future versions |

## 4. Memory Integration System

The "Well" Component: Extending beyond the fixed context window.

| Sub-System | Function |
|---|---|
| 4.1 Rolling Context Buffer | Compresses older conversation turns into compact summary vectors; prevents outright discarding of historical data |
| 4.2 Persistent Retrieval Store | Enables retrieval-style lookups for facts established in prior sessions or earlier in long documents; allows on-demand recall of definitions, notation, and claims without restating them |
| 4.3 Efficiency Safeguards | Leverages GQA and the local/global attention split to ensure memory maintenance does not exponentially increase inference costs |

## 5. Training Data Strategy

| Field | Detail |
|---|---|
| Focus | Academic and Technical Corpora |
| Source Material | Papers, textbooks, technical documentation, structured reference data, long-form expository writing |
| Rationale | Rewards two core model strengths: (1) sustained logical flow across long arguments, (2) precise recall of earlier definitions and complex notation |

## 6. Future Development & Ecosystem Role

| Field | Detail |
|---|---|
| 6.1 Strategic Positioning | River Well is designed as a component model, not a standalone end-to-end product |
| 6.2 Target Architecture — Role | The "Language and Memory" backbone |
| 6.2 Target Architecture — Partner System | A Google-like multimodal architecture (e.g., Gemini-style) handling vision, audio, and tool use |
| 6.2 Target Architecture — Synergy | River Well provides coherent textual reasoning and long-term memory, while the companion model handles multimodal grounding |

## 7. Summary Checklist for Implementation

| Feature | Status/Note |
|---|---|
| Base Model | GPT-2-style Decoder, RMSNorm throughout |
| Attention | Interleaved Local/Global + GQA |
| Activations | SwiGLU-style Gated FFN |
| Memory | Compressed rolling buffer + Retrieval store |
| Training Data | Academic/Technical focus |
| Scaling Path | MoE routing for future large-scale versions |
| End Goal | Language/Memory module within a larger multimodal system |
