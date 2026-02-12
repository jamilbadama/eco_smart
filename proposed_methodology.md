# Proposed Methodology

## 1. System Architecture

The proposed Eco-SMART system is designed as a hierarchical multi-agent framework tailored for automated mental health assessment. The architecture leverages a **Supervisor-Worker** pattern, where a central **Clinical Supervisor** agent orchestrates specialized sub-agents to analyze multimodal patient data and provide clinically grounded interpretations.

### 1.1 Multi-Agent Framework
The core of the system is built using the `deepagents` library, integrating Large Language Models (LLMs) with specialized functional tools.
- **Clinical Supervisor (Orchestrator)**: Powered by `gpt-4o-mini`, this agent acts as the primary interface for clinicians. It decomposes complex queries ("Analyze patient X and explain the risk"), delegates tasks to specialists, and synthesizes technical outputs into empathetic, actionable clinical reports.
- **Diagnostic Specialist**: A dedicated sub-agent responsible for the quantitative analysis of patient sessions. It utilizes the `analyze_session` tool to process raw audio, video, and text data, returning a structured risk assessment and dashboard-ready visualizations (`<dashboard_data>`).
- **Knowledge Specialist**: A retrieval-augmented sub-agent tasked with providing medical context. It uses the `retrieve_guidelines` tool to fetch PHQ-8 scoring rules, clinical risk factors, and definitions of behavioral markers, ensuring the Supervisor's final report is medically accurate.

## 2. Multimodal Feature Extraction

The **Diagnostic Specialist** employs a **MarkerDetector** module to extract behavioral biomarkers across three modalities. This module utilizes specialized pre-trained transformer models to ensure robust feature extraction.

### 2.1 Text Modality (Semantic Analysis)
- **Model**: `rafalposwiata/deproberta-large-depression` (RoBERTa-based).
- **Function**: Analyzes patient transcripts to detect depressive semantic patterns.
- **Output**: Classifies text segments into risk categories (e.g., 'moderate', 'severe', 'not depression') and calculates a "risk inverse" sentiment score ($1.0$ for low risk, $0.0$ for high risk).

### 2.2 Audio Modality (Acoustic & Prosodic Analysis)
- **Emotion Recognition**: Uses `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` to detect affective states (e.g., sadness, anxiety).
- **Prosody/Monotone Detection**: Leverages `facebook/wav2vec2-base-960h` to extract latent audio features. The standard deviation of the last hidden state ($\sigma_{pitch}$) serves as a neural proxy for pitch variability, detecting monotone speech patterns indicative of psychomotor retardation.

### 2.3 Video Modality (Behavioral Analysis)
- **Emotion Recognition**: Uses `dima806/facial_emotions_image_detection` for facial affect recognition.
- **Psychomotor Retardation**: Calculates movement intensity by computing the mean absolute difference between consecutive video feature frames. Reduced movement metrics serve as a proxy for psychomotor retardation.

## 3. Fusion and Triangulation

To synthesize these disparate signals into a coherent clinical assessment, the system employs a two-stage fusion process.

### 3.1 MMFformer-inspired Fusion
The `process_multimodal_fusion` module implements a weighted heuristic fusion strategy:
$$ \text{Risk}_{fusion} = w_t \cdot R_{text} + w_a \cdot R_{audio} + w_v \cdot R_{video} + \text{Bonus}_{agreement} $$
Where:
- $w_t=0.4, w_a=0.3, w_v=0.3$ represent the weights for text, audio, and video modalities respectively.
- **Cross-Modal Agreement**: A bonus score is applied if high clinical text risk coincides with monotone acoustic patterns, reinforcing confidence in the diagnosis.

### 3.2 Clinical Triangulation
The **TriangulationEngine** converges evidence from four distinct sources:
1. **Static AI Models**: Traditional binary classifiers.
2. **Self-Reports**: PHQ-8 scores (if available).
3. **Fusion Assessment**: The output from the MMFformer-inspired fusion.
4. **Specialized Biomarkers**: Specific flags from the text/audio analysis (e.g., "Severe" text label, "Monotone" audio).

The engine generates a "Convergence Summary" (Preliminary vs. Moderate vs. Strong) based on the number of agreeing modalities, providing a transparent audit trail for the clinician.

## 4. Interpretation and Reporting

The **InterpretationEngine** maps the triangulated quantitative scores to standardized clinical risk bands (PHQ-8 equivalent):
- **0-4**: Minimal/No Depression
- **5-9**: Mild
- **10-14**: Moderate
- **15-19**: Moderately Severe
- **20+**: Severe

Finally, the **Clinical Supervisor** constructs the response, embedding the technical findings within a natural language report that explains the *why* behind the score (e.g., "Risk level set to Moderate due to converging signs of linguistic negativity and acoustic flattening").
