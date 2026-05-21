#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# 设置输入和输出目录
input_dir = Path('asset/voices')
output_dir = Path('asset/wavs')

# 创建输出目录
output_dir.mkdir(parents=True, exist_ok=True)

# 支持的音频格式
supported_formats = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']

# 获取所有音频文件
audio_files = []
for ext in supported_formats:
    audio_files.extend(input_dir.glob(f'*{ext}'))

print(f'找到 {len(audio_files)} 个音频文件')

# 逐个转换
success_count = 0
fail_count = 0

for audio_file in audio_files:
    # 生成输出文件名
    output_file = output_dir / f'{audio_file.stem}.wav'
    
    # 构建 ffmpeg 命令
    cmd = [
        'ffmpeg',
        '-i', str(audio_file),
        '-ar', '16000',
        '-ac', '1',
        '-sample_fmt', 's16',
        '-y',  # 覆盖已存在的文件
        str(output_file)
    ]
    
    print(f'正在转换: {audio_file.name} -> {output_file.name}')
    
    try:
        # 执行 ffmpeg 命令，隐藏输出
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        print(f'  ✓ 转换成功')
        success_count += 1
    except subprocess.CalledProcessError as e:
        print(f'  ✗ 转换失败: {e}')
        fail_count += 1

print(f'\n转换完成! 成功: {success_count}, 失败: {fail_count}')
print(f'输出目录: {output_dir.absolute()}')
