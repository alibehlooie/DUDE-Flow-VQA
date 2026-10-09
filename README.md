# DUDE-Flow-VQA: Reading-Order Enriched & Robust Document VQA Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/Domain-Document%20VQA%20%7C%20Layout%20Analysis-blue)](#-dataset-overview)
[![Benchmark](https://img.shields.io/badge/Base%20Dataset-DUDE-orange)](#-dataset-overview)

**DUDE-Flow-VQA** is an enriched Document Visual Question Answering (DocVQA) benchmark built upon the multi-page DUDE dataset. It addresses two pervasive challenges in modern Vision-Language Models (VLMs) and document foundation models:

1. **Arbitrary Reading Orders & Unstructured OCR:** Standard OCR engines produce flat coordinate streams that disrupt sequential reasoning across complex, multi-column pages. DUDE-Flow-VQA structures every document token using an **LLM-judged natural reading flow** (`reading_order_index`) paired with **YOLO-detected spatial structural entities** (`[TEXT]`, `[FIGURE]`, `[CAPTION]`, `[ABANDON]`).
2. **Hallucination & Overconfidence on Unanswerable Inquiries:** Typical DocVQA benchmarks assume an answer always exists in the document. DUDE-Flow-VQA introduces systematically constructed **corrupted, unanswerable queries** labeled across structured failure taxonomies to train models to accurately reject invalid or unsupported queries.

---

## 📥 Datasets & Download Links

Access the complete dataset splits and enriched OCR metadata directly via Google Drive:

### Datasets

| Split | Train | Val | Test |
| :--- | :---: | :---: | :---: |
| **Full** | [link*](https://drive.google.com/drive/folders/1RTefxOg2m4jCxnC9BqegIt3D_GcrPws_) | [link*](https://drive.google.com/drive/folders/1JFwrIDzfTzhYQ5JmScj0FkXcWCWlaO1K) | [link*](https://drive.google.com/drive/folders/1h9ccgdM2010ZI5HTdgcq8TV86j4xNHAv) |
| **OCR Enriched** | [link](https://drive.google.com/drive/folders/1Qa0Y83fYcgRqvSUppe2I6MqVGmIDwJX2) | [link](https://drive.google.com/drive/folders/1JFwrIDzfTzhYQ5JmScj0FkXcWCWlaO1K) | [link](https://drive.google.com/drive/folders/1h9ccgdM2010ZI5HTdgcq8TV86j4xNHAv) |
| **Corrupted QA** | [link](https://drive.google.com/file/d/1a3AEDgt2xvP4ylh6uT9yVxViN-fuZRQm/view?usp=drivesdk) | [link](https://drive.google.com/drive/folders/1JFwrIDzfTzhYQ5JmScj0FkXcWCWlaO1K) | [link](https://drive.google.com/drive/folders/1h9ccgdM2010ZI5HTdgcq8TV86j4xNHAv) |

*\* Full dataset root directory: [Google Drive Root](https://drive.google.com/drive/folders/1JJbfyejq15ZSjwKtmC9lWmxKfHhCnZz2)*

---

## 📊 Dataset Statistics & Corruption Distribution

The unanswerable/corrupted subset ($N = 2,339$) is divided into three primary categories to rigorously evaluate model hallucination and layout-level grounding:

```text
Corruption Type Distribution (N = 2,339)
├── ENTITY  ....................... 1,508 (64.5%)
│   ├── MISCELLANEOUS .............   741 (49.1%)
│   ├── NUMERICAL .................   397 (26.3%)
│   ├── TEMPORAL ..................   144 ( 9.5%)
│   ├── STRUCTURAL ................   113 ( 7.5%)
│   └── LOCATION ..................   112 ( 7.4%)
│
├── LAYOUT  .......................   515 (22.0%)
│   ├── PAGE SPECIFIC .............   485 (94.2%)
│   ├── POSITIONAL ................    27 ( 5.2%)
│   └── MULTI PAGE ................     3 ( 0.6%)
│
└── ELEMENT .......................   316 (13.5%)
    ├── PARAGRAPH .................    56 (17.7%)
    ├── TABLE .....................    53 (16.8%)
    ├── CHART .....................    49 (15.5%)
    ├── FIGURE ....................    33 (10.4%)
    └── MIXED .....................    27 ( 8.5%)
```

### Key Insights:
* **Entity Corruptions (64.5%):** Target entity swaps, non-existent numbers/dates, and subtle entity mutations that prompt models to hallucinate answers.
* **Layout Inconsistencies (22.0%):** Positional questions (e.g., *"What is in the top right box on page 3?"*) where the target component is absent or altered.
* **Element Anomalies (13.5%):** Explicit queries about structural objects like tables, figures, or plots that do not exist or lack the requested attribute.

---

## 🎯 Primary Use Cases

1. **Hallucination Mitigation in DocVQA:**  
   Most standard document QA models force an extractive span even when questions are invalid. DUDE-Flow-VQA serves as a benchmark for training models to output rejection tokens (e.g., `not-answerable`) when evidence is missing.

2. **Sequential Layout & Reading Order Modeling:**  
   Because OCR tokens are ordered via an LLM judge, this dataset allows researchers to evaluate whether transformer-based architectures (LayoutLM, Donut, DocFormers) benefit from pre-sorted reading sequences over coordinate-based heuristics.

3. **Multimodal Grounding for Unexplained Visuals:**  
   YOLO regions indicate visual components (`[FIGURE]`, `[CAPTION]`) and their spatial bounding boxes without providing explicit textual interpretations, allowing VLMs to test cross-modal visual reasoning on complex diagrams, charts, and technical schematics.

---

## 📂 Data Schema & Format

### 1. Classified Q&A Annotation (`dataset_classified.json`)

Each entry pairs a question with ground-truth spans, document references, and corruption labels:

```json
{
  "questionId": "d3cc98e79769668664171a681a0375d4_17aeea15d7958827c4338ba9209dd21a",
  "docId": "d3cc98e79769668664171a681a0375d4",
  "question": "What type of driver's license does Austin B. Ewell III have?",
  "answers": [],
  "answer_type": "not-answerable",
  "answers_page_bounding_boxes": null,
  "answer_page_indices": null,
  "image_paths": [
    "images/d3cc98e79769668664171a681a0375d4_0.jpg"
  ],
  "ocr_paths": [
    "OCR/d3cc98e79769668664171a681a0375d4_0.json"
  ],
  "primary_corruption_type": "ENTITY",
  "corruption_subcategory": "MISCELLANEOUS",
  "corruption_details": {}
}
```

### 2. OCR Enriched Annotations (`OCR_Enriched/`)

Each page file contains normalized bounding coordinates, semantic tags, and relative ordering:

```json
{
  "object0": {
    "type": "plain text",
    "tag": "[TEXT]",
    "bbox": [
      0.0586,
      0.0652,
      0.4668,
      0.0805
    ],
    "text": "Clay County Utility Authority Board of Supervisors",
    "page": 2,
    "position": "Top-Left",
    "original_line_index": 119,
    "reading_order_index": 0
  },
  "object1": {
    "type": "figure",
    "tag": "[FIGURE]",
    "bbox": [
      0.7054,
      0.8520,
      0.9648,
      0.9053
    ],
    "text": "Snap-on",
    "page": 0,
    "position": "Bottom-Right",
    "original_line_index": 0,
    "reading_order_index": 1
  }
}
```

* **`bbox` coordinates:** Normalized as `[x_min, y_min, x_max, y_max]` relative to document page width and height ($0.0$ to $1.0$).
* **`reading_order_index`:** Sequential rank determining reading sequence as resolved by the LLM judge.
* **`tag`:** Semantic type identified via layout detection (`[TEXT]`, `[FIGURE]`, `[CAPTION]`, `[ABANDON]`).

---

## 🚀 Quick Usage Example (PyTorch)

```python
import json
import os
from PIL import Image

def load_doc_sample(sample_entry, base_dir="./data/train"):
    # Load document image
    img_path = os.path.join(base_dir, sample_entry["image_paths"][0])
    image = Image.open(img_path).convert("RGB")
    
    # Load and sort OCR tokens by LLM reading order
    ocr_path = os.path.join(base_dir, sample_entry["ocr_paths"][0])
    with open(ocr_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)
    
    tokens = list(ocr_data.values())
    tokens.sort(key=lambda x: x.get("reading_order_index", 0))
    
    return {
        "image": image,
        "question": sample_entry["question"],
        "answer_type": sample_entry["answer_type"],
        "is_answerable": sample_entry["answer_type"] != "not-answerable",
        "ordered_text": " ".join([t["text"] for t in tokens if t.get("text")])
    }
```

---

## 📜 Citation

If you use DUDE-Flow-VQA in your research, please cite this repository and the original DUDE benchmark:

```bibtex
@misc{dudeflowvqa2026,
  author = {Ali Behlooie},
  title = {DUDE-Flow-VQA: Reading-Order Enriched and Robust Document VQA Benchmark},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/alibehlooie/Dude-flow-vqa}}
}
```
