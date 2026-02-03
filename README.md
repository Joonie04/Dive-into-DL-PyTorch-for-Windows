# Dive-into-DL-PyTorch-for-Windows
动手学深度学习(PyTorch版)的windows版本[持续更新中]

## 项目结构
- codes: 每一章实现的代码, 为实现独立性移除依赖的代码, 每个文件都是一个独立的程序
- dataset: 代码中需要的数据
- docs: 每一章内容的总结说明文档
- downloader: 数据下载器, 用于下载代码中需要的数据

## 环境配置
### 安装PyTorch
#### 安装PyTorch的CPU版本
```bash
pip install torch torchvision
```
#### 安装PyTorch的GPU版本
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
### 安装其他依赖
```bash
pip install matplotlib numpy pandas scikit-learn
```

## 感谢优秀的开源项目
- [动手学深度学习(PyTorch版)](https://zh.d2l.ai/)
- [动手学深度学习(PyTorch版)中文版本](https://github.com/d2l-ai/d2l-zh)
- [动手学深度学习(PyTorch版)英文版本](https://github.com/d2l-ai/d2l-en)
- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch中文文档](https://pytorch-cn.readthedocs.io/zh/latest/index.html)
- 