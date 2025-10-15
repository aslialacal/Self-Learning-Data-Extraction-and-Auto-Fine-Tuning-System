# Self-Learning Data Extraction and Auto Fine-Tuning System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-orange.svg)](https://huggingface.co/docs/transformers)

A sophisticated AI-powered system for automated information extraction from structured and unstructured documents with continuous self-learning capabilities through automated fine-tuning.

## 🌟 Key Features

### 1. **Unified Multi-Format Data Handling**
- ✅ **Structured Data**: CSV, Excel, SQL databases
- ✅ **Unstructured Data**: PDF documents, scanned images (OCR-enabled)
- ✅ **Multi-language Support**: English, German, and more via multilingual models

### 2. **Intelligent QA-Based Extraction**
- 🤖 **Base Model**: XLM-RoBERTa-large-squad2 (deepset)
- 🎯 **Question-Answering Approach**: Natural language questions for flexible extraction
- 📊 **Confidence Scoring**: Every extraction includes reliability metrics
- 🔄 **LoRA Fine-Tuning**: Parameter-efficient adaptation with PEFT

### 3. **Automated Self-Learning Pipeline**
- 📝 **Human-in-the-Loop Feedback**: User corrections automatically logged
- 🔄 **Auto Fine-Tuning**: Triggers retraining after threshold (5+ corrections)
- 📈 **Continuous Improvement**: System learns from every correction
- 💾 **Version Control**: Track model performance across iterations

### 4. **Comprehensive Evaluation Framework**
- 🎯 **Multi-Model Consensus**: 3 QA models for robust validation
- 📊 **Statistical Metrics**: Success rate, accuracy, confidence distributions
- 📈 **Performance Tracking**: Version-based comparison and reporting
- 🔍 **Automated Evaluation**: SmartAutomatedEvaluator with batch processing

### 5. **Multi-Domain Adaptability**
- 💼 **Finance**: Invoices, financial statements, contracts
- 🏥 **Healthcare**: Medical records, lab reports, prescriptions
- ⚖️ **Legal**: Court documents, contracts, legal notices
- 👥 **HR**: Resumes, employee records, performance reviews
- 🎓 **Education**: Transcripts, research papers
- 🛒 **Retail**: Receipts, inventory reports

### 6. **Explainable AI Integration**
- 🔍 **LIME Explanations**: Feature importance for predictions
- 🎨 **Attention Visualization**: Heatmaps showing model focus
- 📊 **Confidence Interpretation**: Trust indicators and reliability scores
- 🎯 **Transparent Predictions**: Every answer comes with explanation

### 7. **Enterprise-Grade Security**
- 🔒 **Encryption**: AES-256 encryption for sensitive data
- 🌐 **Offline Operation**: Complete air-gapped deployment support
- 🔐 **Secure Storage**: Encrypted model and data management
- 📝 **Audit Trail**: Complete security logging and compliance

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
PyTorch 2.0+
CUDA (optional, for GPU acceleration)
Tesseract OCR (for image processing)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/self-learning-data-extraction.git
cd self-learning-data-extraction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Tesseract OCR** (for image processing)
- Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

### Basic Usage

```python
from qa_extractor import QABasedExtractor

# Initialize the extractor
extractor = QABasedExtractor(model_name="deepset/xlm-roberta-large-squad2")

# Extract information from a document
questions = [
    "What is the invoice number?",
    "What is the total amount?",
    "What is the company name?"
]

results = extractor.extract_information(
    text=document_text,
    custom_questions=questions
)

# Access high-confidence extractions
high_conf = extractor.get_high_confidence_extractions(results)
print(high_conf)
```

## 📚 Documentation

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│               Document Ingestion Layer                      │
│  (PDF, CSV, Excel, Images, SQL) → OCR → Text Extraction     │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              QA-Based Extraction Engine                     │
│  XLM-RoBERTa + LoRA Fine-Tuning + Confidence Scoring        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│           Feedback Collection & Curation                    │
│  User Corrections → SQuAD Format → Training Data            │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│          Automated Fine-Tuning Pipeline                     │
│  Threshold Detection → LoRA Training → Model Update         │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│         Evaluation & Version Management                     │
│  Multi-Model Consensus → Metrics → Safe Promotion           │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. QABasedExtractor
Main extraction engine with LoRA fine-tuning support.

#### 2. ModelEvaluator
Comprehensive evaluation framework with multi-model consensus.

#### 3. ModelVersionManager
Tracks model versions and safely promotes improvements.

#### 4. DomainTemplateManager (for future dimensions)
Manages domain-specific question templates for 6+ industries.

#### 5. ExplainableQASystem
LIME + Attention visualization for transparent predictions.

#### 6. SecureOfflineSystem
Enterprise-grade security with encryption and air-gapped operation.


### Performance Metrics

**On 100-invoice test dataset:**
- ✅ Success Rate: 62.5% - 87.9%
- ✅ Mean Consensus Accuracy: 85.3%
- ✅ Reliability Rate: 92.1%

## Notebooks

The project includes comprehensive Jupyter notebooks:

1. **Main System** (`Self-Learning Data Extraction and Auto Fine-Tuning System.ipynb`)
   - Complete implementation
   - Step-by-step execution
   - Training and evaluation

2. **With Explainability** (`Self-Learning Data Extraction and Auto Fine-Tuning System.ipynb`)
   - LIME integration
   - Attention visualization
   - Trust indicators

## 🛠️ Configuration

### LoRA Fine-Tuning Parameters

```python
lora_config = LoraConfig(
    r=16,                    # Rank of LoRA matrices
    lora_alpha=32,          # Scaling factor
    target_modules=[
        "query", "key", "value", "dense"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.QUESTION_ANS
)
```

### Training Arguments

```python
training_args = TrainingArguments(
    output_dir="./lora_fine_tuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_strategy="epoch"
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgments

- **Hugging Face** for the Transformers library and model hub
- **deepset** for the XLM-RoBERTa-large-squad2 model
- **Microsoft** for the LoRA implementation (PEFT)
- **LIME** for explainability framework

## 📧 Contact

For questions, suggestions, or collaborations:
Email: aslialacal@hotmail.com



