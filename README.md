# 中文文本分类系统

一个基于深度学习的中文文本分类系统，支持多种模型架构和训练策略。

## 功能特性

- 支持多种深度学习模型：
  - TextCNN：使用卷积神经网络进行文本分类
  - LSTM：使用长短期记忆网络进行文本分类
  - BERT：使用预训练的中文BERT模型进行文本分类

- 数据处理功能：
  - 文本预处理（URL移除、邮件移除、标点符号处理等）
  - 数据增强（同义词替换、随机删除、随机交换、随机插入）
  - 支持自定义词表和标签映射

- 训练功能：
  - 支持多种优化器（Adam、SGD）
  - 学习率调度
  - 早停机制
  - 模型检查点保存
  - TensorBoard可视化

- 评估功能：
  - 分类报告生成
  - 混淆矩阵可视化
  - ROC曲线分析
  - 错误样本分析
  - 特征重要性分析（适用于可解释模型）

- 配置管理：
  - 支持JSON和YAML格式的配置文件
  - 配置验证和默认值
  - 灵活的配置更新机制

## 项目结构

```
Text_Classification/
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后的数据
│   └── vocab/             # 词表文件
├── models/                 # 模型目录
│   ├── checkpoints/       # 模型检查点
│   └── logs/             # TensorBoard日志
├── evaluation/            # 评估结果
├── src/                   # 源代码
│   ├── config/           # 配置管理
│   ├── data/             # 数据处理
│   ├── models/           # 模型定义
│   ├── train/            # 训练相关
│   ├── evaluate/         # 评估相关
│   └── utils/            # 工具函数
├── tests/                 # 测试代码
├── notebooks/            # Jupyter notebooks
├── requirements.txt      # 项目依赖
└── setup.py             # 安装脚本
```

## 安装说明

1. 克隆项目：
```bash
git clone https://github.com/yourusername/Text_Classification.git
cd Text_Classification
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 安装项目：
```bash
pip install -e .
```

## 使用说明

1. 准备数据：
   - 将训练数据放在 `data/raw/` 目录下
   - 数据格式：每行一个样本，格式为 "标签\t文本"

2. 配置模型：
   - 复制 `configs/default.yaml` 为 `configs/my_config.yaml`
   - 根据需要修改配置参数

3. 训练模型：
```python
from src.config import TextClassifierConfig
from src.train import TextClassifierTrainer
from src.data import CNewsDataset

# 加载配置
config = TextClassifierConfig('configs/my_config.yaml')

# 准备数据
dataset = CNewsDataset(config.get_data_config())

# 创建训练器
trainer = TextClassifierTrainer(
    model_type=config['model']['model_type'],
    model_config=config.get_model_config(),
    dataset=dataset,
    training_config=config.get_training_config()
)

# 训练模型
trainer.train()
```

4. 评估模型：
```python
from src.evaluate import TextClassifierEvaluator

# 创建评估器
evaluator = TextClassifierEvaluator(
    model=trainer.model,
    dataset=dataset,
    config=config.get_evaluation_config()
)

# 评估模型
metrics = evaluator.evaluate(test_data)
```

5. 使用模型预测：
```python
# 预测单个文本
text = "这是一条测试文本"
prediction, probability = evaluator.predict([text])
print(f"预测类别: {prediction[0]}")
print(f"预测概率: {probability[0]}")
```

## 配置说明

配置文件支持JSON和YAML格式，主要包含以下配置节：

- `data`: 数据相关配置
  - `train_file`: 训练数据文件路径
  - `val_file`: 验证数据文件路径
  - `test_file`: 测试数据文件路径
  - `max_length`: 文本最大长度
  - `preprocessing`: 文本预处理选项

- `model`: 模型相关配置
  - `model_type`: 模型类型（textcnn/lstm/bert）
  - `vocab_size`: 词表大小
  - `embedding_dim`: 词向量维度
  - `num_classes`: 类别数量
  - 模型特定配置（如TextCNN的filter_sizes等）

- `training`: 训练相关配置
  - `batch_size`: 批处理大小
  - `epochs`: 训练轮数
  - `learning_rate`: 学习率
  - `optimizer`: 优化器类型
  - 其他训练参数（早停、学习率调度等）

- `augmentation`: 数据增强配置
  - 各种增强方法的参数

- `evaluation`: 评估相关配置
  - 评估指标和可视化选项

## 开发说明

1. 代码风格：
   - 遵循PEP 8规范
   - 使用类型注解
   - 编写详细的文档字符串

2. 测试：
   - 运行单元测试：`pytest tests/`
   - 运行代码覆盖率：`pytest --cov=src tests/`

3. 贡献：
   - Fork项目
   - 创建特性分支
   - 提交更改
   - 发起Pull Request

## 结果
### CNN
速度相当快，效果也不错，precision与recall都趋近于0.9  
![image](https://github.com/sun830910/Text_Classification/blob/master/img/CNN_result.png)


