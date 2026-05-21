# Introduction
+ https://modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
  
## Repository Clone
```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive
```

## UV

```bash
# Using uv (recommended)
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install setuptools==69.5.1
uv pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com --index-strategy unsafe-best-match
```

## Download Model
```python
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('iic/CosyVoice-ttsfrd', local_dir='pretrained_models/CosyVoice-ttsfrd')
```
or
```
python download.py
```

## Unzip Resources
```bash
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
uv pip install ttsfrd_dependency-0.1-py3-none-any.whl
uv pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```


## Execute Example
```bash
python -m inference.generator
```

