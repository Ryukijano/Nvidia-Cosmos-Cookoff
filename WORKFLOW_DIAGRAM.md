# Cosmos Sentinel Workflow Architecture

## System Overview

```mermaid
flowchart TB
    subgraph Input["📥 Input Layer"]
        A[Video Upload<br/>MP4/AVI/MOV] --> B{Input Mode}
        C[RTSP Stream] --> B
        D[Live Camera Feed] --> B
        E[API / Webhook] --> B
    end

    subgraph Pipeline["🔧 Core Pipeline"]
        B --> F[BADAS V-JEPA2<br/>Collision Detection]
        F --> G{Collision<br/>Detected?}
        G -->|No| H[Low Risk Path]
        G -->|Yes| I[Extract Pre-Alert Clip]
        I --> J[Cosmos Reason 2<br/>Risk Narration]
        J --> K{Include<br/>Predict?}
        K -->|No| L[BADAS + Reason Only]
        K -->|Yes| M[Cosmos Predict<br/>Counterfactual Generation]
        M --> N[Prevented Continuation]
        M --> O[Observed Continuation]
    end

    subgraph Analysis["📊 Analysis Layer"]
        F --> P[Gradient Saliency Map]
        F --> Q[Probability Timeline]
        J --> R[Risk Score Gauge]
        J --> S[Incident Classification]
        J --> T[BBOX Visualization]
        N --> U[What-If Scenario A]
        O --> V[What-If Scenario B]
    end

    subgraph Output["📤 Output Layer"]
        H --> W[Safe Status Report]
        L --> X[Risk Assessment Report]
        X --> Y[Incident Log Entry]
        U --> Z[Generated Video A]
        V --> AA[Generated Video B]
        X --> AB[API Response]
        Y --> AC[Webhook Alert]
    end

    subgraph Enterprise["🏢 Enterprise Features"]
        AD[Multi-Camera Grid] --> AE[Stream Manager]
        AE --> AF[Rolling Frame Buffer]
        AF --> AG[Live Risk Scoring]
        AG --> AH[Real-time Dashboard]
        AB --> AI[API Rate Limiting]
        AC --> AJ[Slack/Teams Notify]
    end
```

## Detailed Pipeline Flow

```mermaid
sequenceDiagram
    participant User
    participant API as API Gateway
    participant SM as Stream Manager
    participant BD as BADAS Detector
    participant CE as Clip Extractor
    participant RN as Risk Narrator
    participant CP as Cosmos Predict
    participant DB as Data Store
    participant WH as Webhook Service

    User->>API: Upload Video / Stream URL
    API->>SM: Initialize Stream
    SM->>SM: Rolling Frame Buffer
    
    loop Frame Processing
        SM->>BD: Analyze Frame Batch
        BD->>BD: V-JEPA2 Forward Pass
        BD->>BD: Gradient Saliency
        BD-->>SM: Collision Probability
    end

    alt Collision Detected
        SM->>CE: Extract Pre-Alert Clip
        CE->>CE: FFmpeg Timestamp Crop
        CE-->>RN: Focus Video Segment
        
        RN->>RN: Qwen3VL Processing
        RN->>RN: Prompt Engineering
        RN-->>API: Risk Narration
        
        API->>CP: Generate Counterfactuals
        CP->>CP: Cosmos Diffusion Model
        CP-->>API: Prevented/Observed Videos
        
        API->>DB: Store Results
        API->>WH: Fire Webhook
        WH-->>User: Slack Alert
    else No Collision
        API->>DB: Log Safe Status
    end
    
    API-->>User: Pipeline Results
```

## Data Flow Architecture

```mermaid
flowchart LR
    subgraph Inputs["Input Sources"]
        I1[Static Video]
        I2[RTSP Camera]
        I3[Live Stream]
        I4[API Call]
    end

    subgraph Preprocessing["Preprocessing"]
        P1[Frame Extraction]
        P2[Resize 224x224]
        P3[Normalization]
        P4[Batch Assembly]
    end

    subgraph Models["AI Models"]
        M1[V-JEPA2 Encoder]
        M2[BADAS Detector]
        M3[Qwen3VL Reasoner]
        M4[Cosmos Predict Diffusion]
    end

    subgraph Postprocessing["Postprocessing"]
        PP1[Saliency Maps]
        PP2[Risk Scoring]
        PP3[Video Generation]
        PP4[JSON Payload]
    end

    subgraph Delivery["Delivery"]
        D1[Streamlit UI]
        D2[FastAPI Response]
        D3[Webhook Callback]
        D4[Enterprise Dashboard]
    end

    I1 --> P1
    I2 --> P1
    I3 --> P1
    I4 --> P4

    P1 --> P2 --> P3 --> P4
    P4 --> M1 --> M2
    M2 -->|Collision| M3
    M2 -->|Context| M4
    M3 --> PP2
    M4 --> PP3

    M2 --> PP1
    PP1 --> D1
    PP2 --> PP4
    PP3 --> D1
    PP4 --> D2
    PP4 --> D3
    D2 --> D4
```

## Component Interactions

```mermaid
graph TB
    subgraph Frontend["Frontend Layer"]
        ST[Streamlit Pages]
        ED[Enterprise Dashboard]
        SN[Syndrome-Net QEC]
    end

    subgraph Backend["Backend Layer"]
        API[FastAPI Service]
        SB[Stream Manager]
        PL[Pipeline Runner]
    end

    subgraph AI_Engines["AI Engines"]
        BADAS[BADAS Detector<br/>V-JEPA2 Based]
        REASON[Cosmos Reason 2<br/>Qwen3VL Based]
        PREDICT[Cosmos Predict<br/>Diffusion Based]
    end

    subgraph Infrastructure["Infrastructure"]
        CACHE[Cache Dir<br/>~/.cache/cosmos_sentinel]
        HF_HOME[HuggingFace Hub<br/>Model Downloads]
        GPU[CUDA GPU<br/>A10G/V100]
    end

    subgraph External["External Services"]
        HF[HuggingFace Spaces]
        WEBHOOK[Slack/Teams<br/>Webhooks]
    end

    ST --> API
    ED --> API
    ED --> SB
    SN --> API

    API --> PL
    SB --> PL
    PL --> BADAS
    PL --> REASON
    PL --> PREDICT

    BADAS --> CACHE
    REASON --> HF_HOME
    PREDICT --> GPU

    API --> WEBHOOK
    PL --> HF
```

## Risk Assessment State Machine

```mermaid
stateDiagram-v2
    [*] --> Idle: System Start
    Idle --> Monitoring: Video Input
    
    Monitoring --> Analyzing: Frame Capture
    Analyzing --> LowRisk: P < 0.4
    Analyzing --> MediumRisk: 0.4 ≤ P < 0.65
    Analyzing --> HighRisk: P ≥ 0.65
    
    LowRisk --> Monitoring: Continue
    MediumRisk --> Monitoring: Alert + Continue
    
    HighRisk --> Extracting: Trigger Alert
    Extracting --> Narrating: Clip Ready
    
    Narrating --> Predicting: Risk Confirmed
    Narrating --> Monitoring: False Positive
    
    Predicting --> Reporting: Counterfactuals Generated
    Reporting --> Webhook: Fire Notifications
    Reporting --> Monitoring: Reset
    
    Webhook --> [*]: Alert Delivered
```

## Enterprise API Flow

```mermaid
flowchart TD
    subgraph Client["API Client"]
        C1[HTTP Request]
        C2[X-API-Key Header]
    end

    subgraph Gateway["API Gateway"]
        R1[Rate Limiter<br/>Sliding Window]
        R2[Auth Validator]
        R3[Job Queue]
    end

    subgraph Processing["Async Processing"]
        W1[Background Worker]
        W2[Video Analyzer]
        W3[Frame Analyzer]
    end

    subgraph Events["Event System"]
        E1[High Risk Detector]
        E2[Webhook Dispatcher]
        E3[httpx Async Client]
    end

    C1 -->|POST /analyze/video| R1
    C2 --> R2
    R1 -->|RPM Check| R2
    R2 -->|Valid| R3
    R3 -->|Job ID| C1
    R3 --> W1
    W1 --> W2
    W1 --> W3
    W2 --> E1
    E1 -->|P ≥ threshold| E2
    E2 --> E3
    E3 -->|POST JSON| C4[Slack/Teams]
    W2 -->|Result| R4[Job Store]
    C1 -->|GET /jobs/{id}| R4
```

## Model Architecture Stack

```mermaid
graph BT
    subgraph Foundation["Foundation Models"]
        VJ[V-JEPA2<br/>Self-Supervised Video]
        QW[Qwen3-VL<br/>Vision-Language]
        CD[Cosmos Diffusion<br/>Video Generation]
    end

    subgraph Specialized["Specialized Layers"]
        BAD[BADAS Head<br/>Collision Detection]
        REA[Reason Head<br/>Risk Narration]
        PRE[Predict Head<br/>Counterfactuals]
    end

    subgraph Application["Application Layer"]
        CS[Cosmos Sentinel<br/>Pipeline]
        API[FastAPI Service]
        DASH[Enterprise Dashboard]
    end

    VJ --> BAD
    QW --> REA
    CD --> PRE
    
    BAD --> CS
    REA --> CS
    PRE --> CS
    
    CS --> API
    CS --> DASH
```

---

## Key Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| Video Encoding | FFmpeg, Decord | Frame extraction & preprocessing |
| Vision Model | V-JEPA2 | Spatiotemporal representation learning |
| Language Model | Qwen3-VL | Multimodal risk narration |
| Generation | Cosmos Diffusion | Counterfactual video synthesis |
| API Framework | FastAPI | High-performance async API |
| UI Framework | Streamlit | Interactive dashboard & demos |
| Webhooks | httpx | Async webhook dispatch |
| Rate Limiting | In-Memory Sliding Window | API tier enforcement |
| Streaming | OpenCV, Threading | RTSP/MJPEG live feed handling |

## Deployment Options

```mermaid
flowchart LR
    subgraph Local["Local Development"]
        L1[Docker Compose]
        L2[GPU Workstation]
    end

    subgraph Cloud["Cloud Platforms"]
        C1[HuggingFace Spaces]
        C2[AWS/GCP/Azure GPU]
        C3[Self-Hosted K8s]
    end

    subgraph Edge["Edge Deployment"]
        E1[NVIDIA Jetson]
        E2[Industrial PC]
    end

    L1 --> C1
    L2 --> C2
    C2 --> C3
    C3 --> E1
    C3 --> E2
```
