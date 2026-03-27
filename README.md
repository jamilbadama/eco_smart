# Eco-SMART: Service-Oriented Platform for Mental Health Assessment

Eco-SMART is an advanced AI platform designed to assist clinicians in screening for depression using multimodal data (Audio, Video, and Text). 
---

## 🚀 Key Features

-   **SOTA Multimodal Fusion (MMFformer)**: Neural hub for spatio-temporal detection of psychomotor retardation and prosodic flattening.
-   **Real-time Patient Monitoring**: Live audio/video/text biomarker extraction via WebSocket.
-   **Specialized Clinical Models**: Integrated `deproberta-large-depression` and `Wav2Vec2` for high-precision diagnostic signals.
-   **Interactive Findings Display**: Instant clinical summary modals identifying high-level behavioral patterns.
-   **Layered Service Architecture**: Clearly separated Data, AI Infrastructure, Analytics, Interface, and Services layers.
-   **Triangulation Engine**: Converges multiple modality signals and clinical questionnaires into a unified evidence summary.
-   **Multi-Agent Orchestration**: Specialized agents for diagnostics and clinical knowledge integration.

---

## 🛠 Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd eco-smart
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables**:
    Create a `.env` file in the root:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    DATA_ROOT=./daic_woz_data
    ```

4.  **Data Structure**:
    Ensure your data is organized as follows:
    ```
    daic_woz_data/
    ├── sessions/
    │   ├── 300_P/
    │   ├── 301_P/
    ├── train_split_Depression_AVEC2017.csv
    ├── dev_split_Depression_AVEC2017.csv
    └── test_split_Depression_AVEC2017.csv
    ```

---

## 🚦 How to Run

### 1. Web Application (Recommended)
Launch the platform UI and API:
```bash
python app.py
```
Access the dashboard at `http://localhost:8000`.

### 2. Multi-Agent CLI
Run the clinical supervisor interactively:
```bash
python -m ecosmart.ai_infra.orchestrator
```

---

## 📂 Project Structure

-   `ecosmart/data_layer/`: Ingestion, preprocessing, and session storage.
-   `ecosmart/ai_infra/`: Marker detection, Triangulation, and Interpretation engines.
-   `ecosmart/analytics/`: Behavioral trend tracking and dashboard services.
-   `ecosmart/interface/`: FastAPI documentation and UI backend.
-   `ecosmart/services/`: Cross-cutting concerns like Authentication and Audit logging.
-   `ecosmart/evaluation/`: Training scripts and model performance assessment.

---

---

## 🧠 SOTA Fusion: MMFformer
Eco-SMART implements a state-of-the-art multimodal fusion hub inspired by the **MMFformer (2025)** research architecture. This engine performs high-level spatio-temporal analysis:
- **Psychomotor Retardation**: Cross-modal tracking of reduced visual movement and vocal energy.
- **Affective Mismatch**: Detection of incongruencies between sentiment in speech and facial expressions.
- **Clinical Convergence**: Automated triangulation that flags "Extremely Strong Convergence" when multimodal biomarkers align.

---

## 📝 Outputs & Results

-   **Interactive Clinical Findings**: Post-session summary modals providing instant diagnostic feedback on behavioral patterns.
-   **Clinical Dashboards**: Real-time visualization of risk levels, confidence markers, and behavioral evidence.
-   **Triangulated Evidence**: Synthesized reports explaining "why" a risk level was assigned based on multimodal patterns.
-   **Audit Logs**: All clinical decisions and system accesses are logged for security and transparency.

---

