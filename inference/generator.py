#!/usr/bin/env python3
import sys
import os
import warnings
import argparse
import subprocess

# 检查是否已经是子进程
if os.environ.get('__SILENT_RUN') != '1':
    # 父进程：重定向并以子进程方式运行
    os.environ['__SILENT_RUN'] = '1'
    os.environ['ORT_LOGGING_LEVEL'] = '3'
    os.environ['ORT_LOG_LEVEL'] = '3'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    
    # 作为子进程运行自己
    result = subprocess.run(
        [sys.executable, '-u', __file__] + sys.argv[1:],
        capture_output=True,
        text=True
    )
    
    # 只过滤并打印我们想要的输出
    for line in result.stdout.splitlines():
        if line.startswith('Success,') or line.startswith('Task failed'):
            print(line)
    
    sys.exit(result.returncode)
    

# ===== 下面是实际的业务逻辑 =====


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel
import logging
import torchaudio
import time

# 设置 logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
for logger_name in ['transformers', 'deepspeed', 'onnxruntime', 'torch', 'lightning', 'pkg_resources', 'festival', 'cosyvoice']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

warnings.filterwarnings("ignore")


class SpeechGenerator:
    def __init__(self):
        self.model_dir = 'pretrained_models/Fun-CosyVoice3-0.5B'
        self.cosyvoice = AutoModel(model_dir=self.model_dir)

    def timestamp(self):
        return int(time.time() * 1000)

    def voice_clone(
        self, 
        tts_text, 
        prompt_wav,
        save_file=''
    ):
        '''
        example:
        tts_text = '八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮'
        prompt_wav = './asset/zero_shot_prompt.wav'
        save_file = 'outputs/tt_clone_{}.wav'.format(self.timestamp())
        '''
        prompt_text='You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。'
        if len(save_file) == 0:
            save_file = 'outputs/tt_clone_{}.wav'.format(self.timestamp())
        
        try:
            for _, j in enumerate(
                self.cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav, stream=False)
            ):
                torchaudio.save(save_file, j['tts_speech'], self.cosyvoice.sample_rate)
            print("Success, save file {}".format(save_file))
            
        except Exception as e:
            print("Task failed ", e)
    
    
    def cross_lingual_gen(
        self, 
        tts_text,
        prompt_wav,
        save_file='',
    ):
        '''
        example:
        tts_text = 'You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]'
        prompt_wav = './asset/zero_shot_prompt.wav'
        save_file = 'outputs/tts_cross_{}.wav'.format(self.timestamp())
        '''
        if len(save_file) == 0:
            save_file = 'outputs/tts_cross_{}.wav'.format(self.timestamp())
        
        try:
            input_text = f"You are a helpful assistant.<|endofprompt|>{tts_text} "
            for _, j in enumerate(
                self.cosyvoice.inference_cross_lingual(input_text, prompt_wav, stream=False)
            ):
                torchaudio.save(save_file, j['tts_speech'], self.cosyvoice.sample_rate)
            print("Success, save file {}".format(save_file))
            
        except Exception as e:
            print("Task failed ", e)
            
    
    def instruct_gen(
        self, 
        tts_text,
        instruct_prompt,
        prompt_wav,
        save_file='',
    ):
        '''
        example:
        tts_text = '好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。'
        instruct_prompt = 'You are a helpful assistant. 请用广东话表达。<|endofprompt|>'
        prompt_wav = './asset/zero_shot_prompt.wav'
        save_file = 'outputs/tts_instruct_{}.wav'.format(self.timestamp())
        
        tts_text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
        instruct_prompt = 'You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>'
        prompt_wav = './asset/zero_shot_prompt.wav'
        save_file = 'outputs/tts_instruct_{}.wav'.format(self.timestamp())
        
        '''
        if len(save_file) == 0:
            save_file = 'outputs/tts_instruct_{}.wav'.format(self.timestamp())
        
        try:
            instruction = f"You are a helpful assistant. {instruct_prompt} <|endofprompt|>"
            for _, j in enumerate(
                self.cosyvoice.inference_instruct2(tts_text, instruction, prompt_wav, stream=False)
            ):
                torchaudio.save(save_file, j['tts_speech'], self.cosyvoice.sample_rate)
            print("Success, save file {}".format(save_file))
            
        except Exception as e:
            print("Task failed ", e)


def case():
    generator = SpeechGenerator()
    
    generator.voice_clone('八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮', './asset/zero_shot_prompt.wav')
    generator.cross_lingual_gen(
        'You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]', 
        './asset/zero_shot_prompt.wav'
    )
    generator.instruct_gen(
        '好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。',
        'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
        './asset/zero_shot_prompt.wav'
    )


if __name__ == '__main__':
    # case()
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default='voice_clone', choices=['voice_clone', 'cross_lingual_gen', 'instruct_gen'])
    parser.add_argument('--tts_text', type=str, default='八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮')
    parser.add_argument('--prompt_wav', type=str, default='./asset/zero_shot_prompt.wav')
    parser.add_argument('--instruct_prompt', type=str, default='')
    
    args = parser.parse_args()
    generator = SpeechGenerator()
    if args.task_type == 'voice_clone':
        generator.voice_clone(args.tts_text, args.prompt_wav)
    elif args.task_type == 'cross_lingual_gen':
        generator.cross_lingual_gen(args.tts_text, args.prompt_wav)
    elif args.task_type == 'instruct_gen':
        generator.instruct_gen(args.tts_text, args.instruct_prompt, args.prompt_wav)
    else:
        print("Unknown task type")


# python -m inference.generator --tts_text '今天真是个好日子，困的不行，我再在群里随意发言，我有毒。' --instruct_prompt "用粤语生成" --task_type instruct_gen
# python -m inference.generator --tts_text '今天真是个好日子，困的不行，我再在群里随意发言，我有毒。'  --task_type voice_clone
# python -m inference.generator --tts_text '[四川话]八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮'  --task_type cross_lingual_gen 
