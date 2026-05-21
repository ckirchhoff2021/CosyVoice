import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

""" CosyVoice3 Usage, check https://funaudiollm.github.io/cosyvoice3/ for more details
"""
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
# zero_shot usage
for i, j in enumerate(cosyvoice.inference_zero_shot('八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                                                    './asset/zero_shot_prompt.wav', stream=False)):
    torchaudio.save('outputs/zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L280
for i, j in enumerate(cosyvoice.inference_cross_lingual('You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]',
                                                        './asset/zero_shot_prompt.wav', stream=False)):
    torchaudio.save('outputs/fine_grained_control_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# instruct usage, for supported control, check cosyvoice/utils/common.py#L28
for i, j in enumerate(cosyvoice.inference_instruct2('好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。', 'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
                                                    './asset/zero_shot_prompt.wav', stream=False)):
    torchaudio.save('outputs/instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', 'You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>',
                                                    './asset/zero_shot_prompt.wav', stream=False)):
    torchaudio.save('outputs/instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# hotfix usage
for i, j in enumerate(cosyvoice.inference_zero_shot('高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。', 'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
                                                    './asset/zero_shot_prompt.wav', stream=False)):
    torchaudio.save('outputs/hotfix_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
