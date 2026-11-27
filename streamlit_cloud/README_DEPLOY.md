# 加氢哨兵 Streamlit Cloud 部署包

## 结构

- `streamlit_demo/streamlit_app.py`：主入口
- `streamlit_demo/data/`：15 事件回放数据
- `data/`：test split 子集（R05/R07/R09），供动态 Agent 工具取数
- `tools/`、`pipeline/`、`config/`、`knowledge/`、`artifacts/`：Agent/工具/模型依赖

## 上传

1. 本目录内容推到 GitHub（`streamlit_app.py` 在 `streamlit_demo/` 下）；
2. Streamlit Cloud → New app → 选仓库/分支；
3. Main file path：`streamlit_demo/streamlit_app.py`；
4. Python：3.12（`runtime.txt`）；
5. Secrets：
```toml
DEEPSEEK_API_KEY = "sk-..."
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
```
6. Deploy。

> 不包含 `.streamlit/secrets.toml`；不要把 Key 提交到 GitHub。
> 不包含 openJiuwen（Cloud 安装体积大）；Cloud 上建议只用 DeepSeek API，本地再测 openJiuwen。
