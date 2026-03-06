严格禁止生成任何md文档,除非明确要求

# UV 环境管理指南

本项目使用 [uv](https://docs.astral.sh/uv/) 作为 Python 包管理器和虚拟环境管理工具。

## 常用命令

### 环境管理

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境 (Windows)
.venv\Scripts\activate

# 激活虚拟环境 (Linux/macOS)
source .venv/bin/activate
```

### 依赖管理

```bash
# 安装项目依赖
uv pip install -r requirements.txt

# 安装开发依赖
uv pip install -r requirements-dev.txt

# 安装单个包
uv pip install <package-name>

# 以可编辑模式安装当前项目
uv pip install -e .

# 导出当前环境依赖
uv pip freeze > requirements.txt
```

### 运行命令

```bash
# 使用 uv run 在虚拟环境中执行命令
.venv\Scripts\activate

python -m xxxx.xx.xx
```

## 注意事项

- 确保在项目根目录下执行 uv 命令
- `.venv` 目录已添加到 `.gitignore`，不会被提交到版本控制
- 如遇到依赖冲突，可尝试 `uv pip install --upgrade <package>` 更新特定包
