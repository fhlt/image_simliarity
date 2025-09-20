# Image-Text Similarity Evaluation Tool

This tool uses CLIP models to compute similarity scores between images and their corresponding text descriptions, providing detailed performance metrics for data curation and model training.

## Project Structure

```
metrics/
├── models/                     # Model module
│   ├── __init__.py
│   ├── base_evaluator.py      # Base evaluator class
│   └── clip_evaluator.py      # CLIP evaluator implementation
├── data/                      # Data processing module
│   ├── __init__.py
│   ├── data_loader.py         # Data loader
│   ├── image_downloader.py    # Image downloader
│   └── data_processor.py      # Data processor
├── outputs/                   # Output directory
│   ├── results.csv           # Input data file
│   └── images/               # Downloaded images storage directory
├── main.py                   # Main program entry point
├── config.py                 # Configuration system
├── requirements.txt          # Python dependencies
└── README.md                # Project documentation
```

## Features

- **Modular Design**: Clear module separation for easy extension of new evaluation metrics
- **Multiple Model Support**: Supports both CLIP and BLIP models for different use cases
- **CLIP Model Integration**: Uses OpenAI's CLIP model for general image-text similarity computation
- **BLIP Model Integration**: Uses Salesforce's BLIP model for specialized image-text retrieval
- **Batch Processing**: Supports batch processing of large image-text pairs
- **Performance Monitoring**: Records processing time, memory usage, GPU utilization, and other key metrics
- **Error Handling**: Comprehensive error handling and logging mechanisms
- **Configuration System**: Flexible configuration system supporting different scenarios
- **Scalability**: Designed with large-scale data processing scalability in mind

## Installation and Usage

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Quick Start

1. **Run with default configuration**:
   ```bash
   python main.py
   ```

2. **Create configuration file**:
   ```bash
   python main.py --create-config
   ```

3. **Use custom parameters**:
   ```bash
   python main.py --input outputs/results.csv --output outputs/results_with_scores.csv --batch-size 16
   ```

### Model Selection

```bash
# Use CLIP model (default)
python main.py --model CLIP

# Use BLIP model
python main.py --model BLIP

# List available models
python main.py --list-models

# Interactive model selection
python run.py
```

### Advanced Usage

```bash
# Skip image download (if images already exist)
python main.py --no-download

# Skip image validation
python main.py --no-validate

# Use CPU mode
python main.py --device cpu

# Use different CLIP model
python main.py --model CLIP --model-name openai/clip-vit-large-patch14

# Use different BLIP model
python main.py --model BLIP --model-name Salesforce/blip-itm-large-coco
```

## Configuration System

The project uses JSON configuration files to manage various parameters:

```json
{
  "model": {
    "name": "CLIP",
    "model_name": "openai/clip-vit-base-patch32",
    "device": null,
    "batch_size": 32,
    "image_size": 224
  },
  "data": {
    "input_csv": "outputs/results.csv",
    "output_csv": "outputs/results_with_scores.csv",
    "image_dir": "outputs/images",
    "download_images": true,
    "validate_images": true,
    "min_image_width": 64,
    "max_image_width": 4096,
    "min_image_height": 64,
    "max_image_height": 4096,
    "max_image_size_mb": 50.0
  },
  "download": {
    "timeout": 30,
    "max_retries": 3,
    "delay": 0.1,
    "progress_bar": true
  },
  "logging": {
    "level": "INFO",
    "log_file": "evaluation.log",
    "console_output": true
  }
}
```

## Output Files

### 1. Results CSV File
Contains original data plus the following new columns:
- `similarity_score`: Similarity score (0-1)
- `processing_time`: Processing time per image
- `status`: Processing status (success/error/preprocessing_failed)

### 2. Statistics File
Contains detailed statistics:
- Total records and processing results
- Similarity score statistics
- Quality grade distribution

### 3. Performance Metrics File
Contains detailed performance statistics:
- Model information
- Processing statistics
- Performance metrics
- Resource usage

### 4. Log File
Detailed processing logs including error messages and debug information

## Extending New Evaluation Metrics

The project uses a modular design that makes it easy to add new evaluation metrics:

### 1. Create New Evaluator

Create a new evaluator class in the `models/` directory:

```python
# models/new_evaluator.py
from .base_evaluator import BaseSimilarityEvaluator

class NewSimilarityEvaluator(BaseSimilarityEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize specific model
    
    def load_model(self):
        # Load specific model
        pass
    
    def compute_similarity(self, image_path, text):
        # Implement specific similarity computation
        pass
```

### 2. Update Model Registration

Register the new model in `models/__init__.py`:

```python
from .new_evaluator import NewSimilarityEvaluator

__all__ = ['CLIPSimilarityEvaluator', 'BaseSimilarityEvaluator', 'NewSimilarityEvaluator']
```

### 3. Update Configuration System

Add new model configuration options in `config.py`:

```python
@dataclass
class ModelConfig:
    name: str = "CLIP"  # Support "CLIP", "NEW_MODEL"
    # ... other configurations
```

### 4. Update Main Program

Add support for the new model in the `_create_model` method in `main.py`:

```python
def _create_model(self):
    if self.config.model.name.upper() == "CLIP":
        return CLIPSimilarityEvaluator(...)
    elif self.config.model.name.upper() == "NEW_MODEL":
        return NewSimilarityEvaluator(...)
    else:
        raise ValueError(f"Unsupported model type: {self.config.model.name}")
```

## Design Considerations

### 1. Extensibility Design

**Modular Architecture**: Clear module separation where each module handles specific functionality
- `models/`: Evaluation model implementations
- `data/`: Data processing and loading
- `config.py`: Configuration management
- `main.py`: Main program logic

**Plugin Design**: Easy addition of new evaluators through base class inheritance

**Configuration-Driven**: Manage all parameters through configuration files without code changes

### 2. Performance Optimization

**GPU Acceleration**: Automatic detection and use of GPU acceleration
**Batch Processing**: Support for batch processing to improve efficiency
**Memory Management**: Timely release of unnecessary resources
**Error Recovery**: Single image processing failure doesn't affect the entire batch

### 3. Data Curation Applications

**Similarity Thresholds**: Can set thresholds to filter high-quality image-text pairs
**Quality Grading**: Grade data based on similarity scores
- Excellent (≥0.8): High-quality training data
- Good (0.6-0.8): Usable training data
- Fair (0.4-0.6): Requires manual review
- Poor (<0.4): Recommended for removal

## Large-Scale Deployment Considerations

### 1. Distributed Processing
- Use multi-GPU parallel processing
- Implement data sharding and load balancing
- Support distributed file systems

### 2. Storage Optimization
- Implement image caching mechanisms
- Use compressed storage formats
- Support cloud storage integration

### 3. Monitoring and Alerting
- Real-time performance monitoring
- Anomaly detection and alerting
- Processing progress tracking

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use CPU mode
   - Clear GPU cache

2. **Image Download Failures**
   - Check network connection
   - Increase timeout duration
   - Use proxy server

3. **Model Loading Failures**
   - Check network connection
   - Clear model cache
   - Use local model files

## Performance Benchmarks

Performance on standard hardware:
- **GPU**: NVIDIA RTX 3080
- **CPU**: Intel i7-10700K
- **Memory**: 32GB RAM

Average performance for processing 1000 images:
- Processing time: ~2-3 minutes
- Memory usage: ~4-6GB
- GPU utilization: ~80-90%
