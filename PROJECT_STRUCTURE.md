# 项目结构说明

## 重新组织后的项目结构

```
metrics/
├── models/                     # 模型模块
│   ├── __init__.py            # 模块初始化
│   ├── base_evaluator.py      # 基础评估器抽象类
│   └── clip_evaluator.py      # CLIP评估器实现
├── data/                      # 数据处理模块
│   ├── __init__.py            # 模块初始化
│   ├── data_loader.py         # CSV数据加载器
│   ├── image_downloader.py    # 图像下载器
│   └── data_processor.py      # 图像数据处理器
├── outputs/                   # 输出目录
│   ├── results.csv           # 输入数据文件
│   └── images/               # 下载的图像存储目录
├── main.py                   # 主程序入口
├── config.py                 # 配置系统
├── requirements.txt          # Python依赖
├── README.md                # 项目说明
├── config_example.json      # 示例配置文件
├── run.py                   # 快速启动脚本
├── test_structure.py        # 项目结构测试脚本
└── PROJECT_STRUCTURE.md     # 项目结构说明
```

## 模块设计说明

### 1. models/ 模块
**目的**: 实现各种相似度评估模型

**文件说明**:
- `base_evaluator.py`: 定义评估器的基类接口，包含通用的处理逻辑
- `clip_evaluator.py`: CLIP模型的实现，继承自基类
- `__init__.py`: 模块导出，便于其他模块导入

**扩展方式**: 通过继承 `BaseSimilarityEvaluator` 类，实现 `load_model()` 和 `compute_similarity()` 方法

### 2. data/ 模块
**目的**: 处理数据的加载、下载和预处理

**文件说明**:
- `data_loader.py`: 负责CSV文件的加载和验证
- `image_downloader.py`: 负责从URL下载图像文件
- `data_processor.py`: 负责图像文件的验证和预处理
- `__init__.py`: 模块导出

**功能特点**:
- 支持批量下载图像
- 图像格式和尺寸验证
- 错误处理和重试机制
- 进度条显示

### 3. 配置系统 (config.py)
**目的**: 统一管理所有配置参数

**特点**:
- 使用dataclass定义配置结构
- 支持JSON格式配置文件
- 支持命令行参数覆盖
- 类型安全的配置管理

### 4. 主程序 (main.py)
**目的**: 协调各个模块，实现完整的评估流程

**流程**:
1. 加载配置
2. 初始化各个模块
3. 加载和验证数据
4. 下载图像（可选）
5. 运行相似度评估
6. 保存结果和统计信息

## 扩展新指标的步骤

### 1. 创建新的评估器
在 `models/` 目录下创建新的评估器文件：

```python
# models/new_evaluator.py
from .base_evaluator import BaseSimilarityEvaluator

class NewSimilarityEvaluator(BaseSimilarityEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化特定参数
    
    def load_model(self):
        # 加载特定模型
        pass
    
    def compute_similarity(self, image_path, text):
        # 实现特定的相似度计算
        pass
```

### 2. 更新模块导出
在 `models/__init__.py` 中添加新评估器：

```python
from .new_evaluator import NewSimilarityEvaluator

__all__ = ['CLIPSimilarityEvaluator', 'BaseSimilarityEvaluator', 'NewSimilarityEvaluator']
```

### 3. 更新配置系统
在 `config.py` 中支持新模型：

```python
@dataclass
class ModelConfig:
    name: str = "CLIP"  # 支持 "CLIP", "NEW_MODEL"
    # 添加新模型特定的配置参数
    new_model_param: str = "default_value"
```

### 4. 更新主程序
在 `main.py` 的 `_create_model` 方法中添加新模型支持：

```python
def _create_model(self):
    if self.config.model.name.upper() == "CLIP":
        return CLIPSimilarityEvaluator(...)
    elif self.config.model.name.upper() == "NEW_MODEL":
        return NewSimilarityEvaluator(...)
    else:
        raise ValueError(f"不支持的模型类型: {self.config.model.name}")
```

## 使用方式

### 1. 基本使用
```bash
# 使用默认配置
python main.py

# 创建配置文件
python main.py --create-config

# 使用自定义参数
python main.py --input data.csv --output results.csv --batch-size 16
```

### 2. 快速启动
```bash
python run.py
```

### 3. 测试项目结构
```bash
python test_structure.py
```

## 优势

1. **模块化设计**: 清晰的职责分离，便于维护和扩展
2. **可扩展性**: 通过继承基类轻松添加新的评估指标
3. **配置驱动**: 通过配置文件管理参数，无需修改代码
4. **错误处理**: 完善的错误处理和日志记录
5. **性能监控**: 详细的性能指标记录
6. **易于测试**: 每个模块都可以独立测试

## 未来扩展方向

1. **更多评估模型**: BLIP、DALL-E、Flamingo等
2. **分布式处理**: 支持多GPU和多机器并行
3. **Web界面**: 提供可视化的操作界面
4. **API服务**: 提供RESTful API接口
5. **实时处理**: 支持流式数据处理
6. **云集成**: 支持云存储和云计算服务
