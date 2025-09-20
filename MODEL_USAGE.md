# Model Usage Guide

## Supported Models

The tool now supports two different models for image-text similarity evaluation:

### 1. CLIP (OpenAI)
- **Purpose**: General-purpose image-text similarity computation
- **Models**: 
  - `openai/clip-vit-base-patch32` (default) - Faster, good balance
  - `openai/clip-vit-large-patch14` - More accurate, slower
- **Image Size**: 224x224
- **Best for**: Large-scale data processing, general similarity tasks

### 2. BLIP (Salesforce)
- **Purpose**: Specialized for image-text retrieval tasks
- **Models**:
  - `Salesforce/blip-itm-base-coco` (default) - Base model
  - `Salesforce/blip-itm-large-coco` - Large model, more accurate
- **Image Size**: 384x384
- **Best for**: High-quality evaluation, semantic understanding

## Usage Examples

### Command Line Usage

```bash
# Use CLIP model (default)
python main.py --model CLIP

# Use BLIP model
python main.py --model BLIP

# List available models
python main.py --list-models

# Use specific model with custom parameters
python main.py --model CLIP --batch-size 16 --device cpu
python main.py --model BLIP --batch-size 8 --device cuda
```

### Interactive Usage

```bash
# Interactive model selection
python run.py
```

### Programmatic Usage

```python
from config import get_clip_config, get_blip_config
from models import CLIPSimilarityEvaluator, BLIPSimilarityEvaluator

# CLIP evaluator
clip_evaluator = CLIPSimilarityEvaluator(
    model_name="openai/clip-vit-base-patch32",
    device="cuda",
    batch_size=32
)

# BLIP evaluator
blip_evaluator = BLIPSimilarityEvaluator(
    model_name="Salesforce/blip-itm-base-coco",
    device="cuda",
    batch_size=16
)
```

## Model Comparison

| Feature | CLIP | BLIP |
|---------|------|------|
| **Speed** | Fast | Slower |
| **Accuracy** | Good | Better |
| **Memory Usage** | Lower | Higher |
| **Image Size** | 224x224 | 384x384 |
| **Use Case** | General similarity | Text retrieval |
| **Batch Size** | 32+ | 16-32 |

## Configuration

### CLIP Configuration
```json
{
  "model": {
    "name": "CLIP",
    "model_name": "openai/clip-vit-base-patch32",
    "image_size": 224,
    "batch_size": 32
  }
}
```

### BLIP Configuration
```json
{
  "model": {
    "name": "BLIP",
    "model_name": "Salesforce/blip-itm-base-coco",
    "image_size": 384,
    "batch_size": 16
  }
}
```

## Performance Tips

### For CLIP:
- Use larger batch sizes (32-64) for better GPU utilization
- Prefer GPU over CPU for significant speedup
- Consider openai/clip-vit-large-patch14 for better accuracy if speed is not critical

### For BLIP:
- Use smaller batch sizes (8-16) due to higher memory usage
- 384x384 image size provides better accuracy
- Large model variant for best results

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory with BLIP**:
   - Reduce batch size to 8 or 4
   - Use CPU mode: `--device cpu`
   - Use smaller image size (though this reduces accuracy)

2. **Model Download Failures**:
   - Check internet connection
   - Clear Hugging Face cache: `rm -rf ~/.cache/huggingface/`
   - Use VPN if in restricted regions

3. **Slow Performance**:
   - Use GPU if available
   - Increase batch size (for CLIP)
   - Use smaller models for faster processing

## Example Outputs

### CLIP Results:
```
相似度分数统计:
  平均分数: 0.7234
  最高分数: 0.9456
  最低分数: 0.1234
  标准差: 0.1876

质量分级:
  优秀 (≥0.8): 45 (32.1%)
  良好 (0.6-0.8): 67 (47.9%)
  一般 (0.4-0.6): 23 (16.4%)
  较差 (<0.4): 5 (3.6%)
```

### BLIP Results:
```
相似度分数统计:
  平均分数: 0.6789
  最高分数: 0.9123
  最低分数: 0.0987
  标准差: 0.2134

质量分级:
  优秀 (≥0.8): 38 (27.1%)
  良好 (0.6-0.8): 72 (51.4%)
  一般 (0.4-0.6): 25 (17.9%)
  较差 (<0.4): 5 (3.6%)
```

## Choosing the Right Model

### Use CLIP when:
- Processing large datasets
- Speed is important
- General similarity evaluation
- Limited computational resources

### Use BLIP when:
- Quality is more important than speed
- Semantic understanding is crucial
- Image-text retrieval tasks
- Sufficient computational resources
