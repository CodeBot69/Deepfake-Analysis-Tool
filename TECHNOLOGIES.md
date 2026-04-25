# Aegis-Vision — Technology Stack

A complete list of all technologies, libraries, and tools used in this project.

---

## 🖥️ Backend

### Runtime & Language
| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core backend language |

### Web Framework & Server
| Technology | Version | Purpose |
|---|---|---|
| FastAPI | 0.115.0 | Async REST API framework |
| Uvicorn | 0.30.0 | ASGI server (with `standard` extras) |
| Pydantic | 2.9.0 | Data validation and settings management |
| python-multipart | 0.0.9 | Multipart file upload support |
| aiofiles | 24.1.0 | Async file I/O |

### Deep Learning
| Technology | Version | Purpose |
|---|---|---|
| PyTorch (`torch`) | 2.4.0 | Core deep learning framework |
| TorchVision | 0.19.0 | Vision model utilities and transforms |
| timm | 1.0.9 | EfficientNet-V2 and other pretrained model hub |

### Computer Vision & Image Processing
| Technology | Version | Purpose |
|---|---|---|
| OpenCV (`opencv-python-headless`) | 4.10.0.84 | Frame extraction, face detection, image decoding |
| Pillow | 10.4.0 | Image loading and manipulation |
| MediaPipe | 0.10.14 | Face landmark detection, lip-sync analysis |

### Audio Processing
| Technology | Version | Purpose |
|---|---|---|
| librosa | 0.10.2 | Audio feature extraction (MFCC, spectral analysis) |
| soundfile | 0.12.1 | Audio file I/O |

### Scientific Computing
| Technology | Version | Purpose |
|---|---|---|
| NumPy | 1.26.4 | Numerical operations and array processing |
| SciPy | 1.14.0 | Signal processing (FFT, frequency domain analysis) |

### Visualization
| Technology | Version | Purpose |
|---|---|---|
| Matplotlib | 3.9.0 | Grad-CAM heatmap generation and colormap rendering |

### Utilities
| Technology | Version | Purpose |
|---|---|---|
| uuid6 | 2024.7.10 | UUID generation for job and file IDs |

---

## 🌐 Frontend

### Runtime & Language
| Technology | Version | Purpose |
|---|---|---|
| Node.js | 20+ | JavaScript runtime |
| TypeScript | ^5 | Statically typed JavaScript |

### Framework & UI
| Technology | Version | Purpose |
|---|---|---|
| Next.js | 16.2.0 | React-based full-stack web framework (App Router) |
| React | 19.2.4 | Component-based UI library |
| React DOM | 19.2.4 | React DOM renderer |
| CSS Modules | — | Scoped component-level styling |

### Dev Tools
| Technology | Version | Purpose |
|---|---|---|
| ESLint | ^9 | JavaScript/TypeScript linting |
| eslint-config-next | 16.2.0 | Next.js ESLint ruleset |
| @types/node | ^20 | Node.js TypeScript definitions |
| @types/react | ^19 | React TypeScript definitions |
| @types/react-dom | ^19 | React DOM TypeScript definitions |

---

## 🧪 Testing

| Technology | Version | Purpose |
|---|---|---|
| pytest | 8.3.0 | Python test framework |
| httpx | 0.27.0 | Async HTTP client for FastAPI endpoint testing |

---

## 🤖 Detection Models & Algorithms

| Model / Algorithm | Category | Purpose |
|---|---|---|
| EfficientNet-V2 (via `timm`) | CNN | Spatial artifact detection in image frames |
| FFT (Fast Fourier Transform) | Signal Processing | Frequency domain analysis to detect GAN artifacts |
| Grad-CAM | Explainability | Heatmap generation to highlight manipulated regions |
| MediaPipe Face Mesh | Face Analysis | Lip landmark tracking for audio-visual sync detection |
| MFCC (Mel-Frequency Cepstral Coefficients) | Audio Analysis | Audio feature extraction for lip-sync scoring |
| Ensemble Aggregation | Meta-model | Weighted combination of spatial, frequency, and lip-sync scores |

---

## 🗂️ Project Structure Overview

```
Project_deepfake/
├── app.py               # FastAPI backend entry point
├── config.py            # Global configuration and paths
├── requirements.txt     # Python dependencies
├── models/
│   ├── spatial_detector.py    # EfficientNet-V2 spatial analysis
│   ├── frequency_detector.py  # FFT-based frequency analysis
│   ├── lipsync_detector.py    # Audio-visual lip-sync detection
│   └── ensemble.py            # Score aggregation (frame & video level)
├── utils/
│   ├── video_processor.py     # Frame extraction, face detection
│   └── heatmap_generator.py   # Grad-CAM heatmap rendering
├── tests/
│   └── test_api.py            # API endpoint test suite
└── frontend/                  # Next.js frontend application
    ├── src/app/               # App Router pages and components
    └── package.json           # JS dependencies
```
