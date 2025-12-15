import re
import gradio as gr
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Literal, List, Tuple
import sys
import importlib.util
from datetime import datetime

import torch
import numpy as np  
import random    
import s3tokenizer

from soulxpodcast.models.soulxpodcast import SoulXPodcast
from soulxpodcast.config import Config, SoulXPodcastLLMConfig, SamplingParams
from soulxpodcast.utils.dataloader import (
    PodcastInferHandler,
    SPK_DICT, TEXT_START, TEXT_END, AUDIO_START, TASK_PODCAST
)


S1_PROMPT_WAV = "example/audios/female_mandarin.wav"  
S2_PROMPT_WAV = "example/audios/male_mandarin.wav"  

def load_dialect_prompt_data():
    """
    加载方言提示文本文件并格式化为嵌套字典。
    返回结构: {dialect_key: {display_name: full_text, ...}, ...}
    """
    dialect_data = {}
    
    dialect_files = [
        ("sichuan", "example/dialect_prompt/sichuan.txt", "<|Sichuan|>"),
        ("yueyu", "example/dialect_prompt/yueyu.txt", "<|Yue|>"),
        ("henan", "example/dialect_prompt/henan.txt", "<|Henan|>"),
    ]
    
    for key, file_path, prefix in dialect_files:
        dialect_data[key] = {"(无)": ""} 
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    line = line.strip()
                    if line:
                        full_text = f"{prefix}{line}"
                        display_name = f"例{i+1}: {line[:20]}..."
                        dialect_data[key][display_name] = full_text
        except FileNotFoundError:
            print(f"[WARNING] 方言文件未找到: {file_path}")
        except Exception as e:
            print(f"[WARNING] 读取方言文件失败 {file_path}: {e}")
            
    return dialect_data

DIALECT_PROMPT_DATA = load_dialect_prompt_data()
DIALECT_CHOICES = ["(无)", "sichuan", "yueyu", "henan"]


EXAMPLES_LIST = [
    [
        "example/audios/trump.wav",
        """We're in a failing nation. It's a nation in decline. I'm gonna turn it around. Don't wait to vote. Make sure you're registered and cast your vote as soon as you can. God bless you and God bless our beautiful country. Thank you very much.""",
        "example/audios/musk.wav",
        "Uh, so I- I thought- I was trying to think what is the most useful thing that I could... What could I say that could actually be helpful or useful to you in the future? Um, and, uh, I thought, uh, perhaps, uh, tell the story of, um, how I- how I sort of came to be here. How did some of these things happen? And- and maybe there's some lessons there. Um, 'cause I- I often find myself wondering how did this happen.""",
        """[S1] Elon, man, Starship blows my mind. That Mars dream in '26... it's gonna be epic! Happy New Year, brother! 
[S2] Thanks, sir—and Happy New Year! Late '26, sending five uncrewed Starships. Heart's racing just thinking about them touching down on Mars. [S1] <|laughter|> Five ships heading to the red planet? Chills, brother. Pure chills. America leading the way—makes me so proud! 
[S2] Me too. If they land safe, we're one huge step closer to humans there. Feels unreal sometimes. 
[S1] Unreal and beautiful. Planting that flag... tears in my eyes already. 
[S2] American flag first, always. Orbital refueling demos in '26 gotta work—nerves are high, but the excitement's higher. 
[S1] You'll crush it. Starlink connecting Mars? That's domination—and it feels damn good. 
[S2] <|laughter|> Hell yeah. Optimus bots building the future up there—just imagine that. Gives me goosebumps.
[S1] Robots on Mars, no quitting, just grinding. So inspiring. We're not just winning—we're changing humanity forever! 
[S2] Forever. Can't wait, sir. This is everything. Here's to the future!""",
        "example/audios/sample1.wav",
    ],
    [
        "example/audios/wilson.wav",
        "Wow, Elon Musk has co-founded so many incredible companies—like the revolutionary Tesla, the bold SpaceX, and the cutting-edge xAI! He owns around 15% of Tesla—a company he first backed in 2004 and has led as CEO since 2008, pushing it to amazing heights!",
        "example/audios/musk.wav",
        "Uh, so I- I thought- I was trying to think what is the most useful thing that I could... What could I say that could actually be helpful or useful to you in the future? Um, and, uh, I thought, uh, perhaps, uh, tell the story of, um, how I- how I sort of came to be here. How did some of these things happen? And- and maybe there's some lessons there. Um, 'cause I- I often find myself wondering how did this happen.",
        """[S1] Hey Elon, wow, everyone’s buzzing about Optimus in 2026. Will you really produce millions of them?
[S2] Oh yeah, once we’re at full scale, the plan is millions.
[S1] Awesome! And the price — will it hit around $20k?
[S2] Yep, absolutely under $20k at high volume to make it affordable for pretty much everyone.
[S1] Nice! Are the Gen 3 hands with 50 actuators the biggest upgrade?
[S2] Definitely, man. The dexterity is next level — way better than human hands.
[S1] Whoa, it’s already folding laundry and cooking, right? How much is real autonomy vs remote control these days?
[S2] Haha, it’s doing laundry, cooking, even yoga. Core tasks are pure end-to-end AI; we only use remote control for safety in some demos.
[S1] Cool! When you start selling — factories first or homes? And one going to Mars in 2026?
[S2] Factories first, then homes. And yes, at least one on Starship to Mars. Early Merry Christmas!""",
        "example/audios/sample2.wav",
    ],
    [
        S1_PROMPT_WAV,
        "喜欢攀岩、徒步、滑雪的语言爱好者，以及过两天要带着全部家当去景德镇做陶瓷的白日梦想家。",
        S2_PROMPT_WAV,
        "呃，还有一个就是要跟大家纠正一点，就是我们在看电影的时候，尤其是游戏玩家，看电影的时候，在看到那个到西北那边的这个陕北民谣，嗯，这个可能在想，哎，是不是他是受到了黑神话的启发？",
        """[S1] 亲爱的听众们，哈喽哈喽，圣诞快乐啊！欢迎收听我们的科技播客《希强聊车》！我是小希。
[S2] 我是小强。圣诞快乐啊！今天聊聊特斯拉FSD在中国啥情况？小希，你老车主了，有啥新动态？
[S1] 哎呀，当年咬牙加了6.4万FSD，等得我心碎啊！好在今年2月终于推送城市智能辅助了，自动变道、认红绿灯挺聪明，最近更新后更顺手，我都想夸它“宝贝真棒”呀！
[S2] 心动了<|laughter|>！现在能完全放手吗？像美国那样？
[S1] 还不行呢～还是L2级别，得盯着路、手放方向盘。中国版跟美国有差距，比如不能全自动停车位进出。复杂路口偶尔犹豫，但整体用着越来越舒服了，真的！
[S2] 为什么推进这么谨慎呢？马斯克不是一直说很快吗？
[S1] 哈哈，主要监管和数据安全嘛，数据必须本地存。上海数据中心建好后才慢慢解锁，今年试用活动也没多久就暂停了。嗯，安全第一，我完全理解啦！
[S2] 未来能追上国产智驾吗？小鹏华为现在多爽啊。
[S1]  哇，竞争超级激烈呢！国产迭代快，但特斯拉纯视觉端到端潜力大。我超看好哦！马斯克11月说部分批准了，2026年2-3月估计完全放开，到时候功能接近美国版，销量肯定涨啦！
[S2] 还得熬几个月。老车主急眼了吧？
[S1] 是啊，尤其是HW3的，心里不是滋味。我HW4还行。总之推进虽慢，但稳，安全优先。完全放开后，特斯拉在中国有戏！
[S2]  对，大市场不能掉队。好了，今天到这儿。圣诞快乐，新年快乐！
[S1] 圣诞快乐～大家节日愉快！下期见啦！""",
        "example/audios/sample3.wav",
    ],
]


model: SoulXPodcast = None
dataset: PodcastInferHandler = None
def initiate_model(config: Config, enable_tn: bool=False):
    global model
    if model is None:
        model = SoulXPodcast(config)

    global dataset
    if dataset is None:
        dataset = PodcastInferHandler(model.llm.tokenizer, None, config)

_i18n_key2lang_dict = dict(
    # Speaker1 Prompt
    spk1_prompt_audio_label=dict(
        en="Speaker 1 Prompt Audio",
        zh="说话人 1 参考语音",
    ),
    spk1_prompt_text_label=dict(
        en="Speaker 1 Prompt Text",
        zh="说话人 1 参考文本",
    ),
    spk1_prompt_text_placeholder=dict(
        en="text of speaker 1 Prompt audio.",
        zh="说话人 1 参考文本",
    ),
    spk1_dialect_prompt_text_label=dict(
        en="Speaker 1 Dialect Prompt Text",
        zh="说话人 1 方言提示文本",
    ),
    spk1_dialect_prompt_text_placeholder=dict(
        en="Dialect prompt text with prefix: <|Sichuan|>/<|Yue|>/<|Henan|> ",
        zh="带前缀方言提示词思维链文本，前缀如下：<|Sichuan|>/<|Yue|>/<|Henan|>，如：<|Sichuan|>走嘛，切吃那家新开的麻辣烫，听别个说味道硬是霸道得很，好吃到不摆了，去晚了还得排队！",
    ),
    # Speaker2 Prompt
    spk2_prompt_audio_label=dict(
        en="Speaker 2 Prompt Audio",
        zh="说话人 2 参考语音",
    ),
    spk2_prompt_text_label=dict(
        en="Speaker 2 Prompt Text",
        zh="说话人 2 参考文本",
    ),
    spk2_prompt_text_placeholder=dict(
        en="text of speaker 2 prompt audio.",
        zh="说话人 2 参考文本",
    ),
    spk2_dialect_prompt_text_label=dict(
        en="Speaker 2 Dialect Prompt Text",
        zh="说话人 2 方言提示文本",
    ),
    spk2_dialect_prompt_text_placeholder=dict(
        en="Dialect prompt text with prefix: <|Sichuan|>/<|Yue|>/<|Henan|> ",
        zh="带前缀方言提示词思维链文本，前缀如下：<|Sichuan|>/<|Yue|>/<|Henan|>，如：<|Sichuan|>走嘛，切吃那家新开的麻辣烫，听别个说味道硬是霸道得很，好吃到不摆了，去晚了还得排队！",
    ),
    # Dialogue input textbox
    dialogue_text_input_label=dict(
        en="Dialogue Text Input",
        zh="合成文本输入",
    ),
    dialogue_text_input_placeholder=dict(
        en="[S1]text\n[S2]text\n[S1]text...",
        zh="[S1]文本\n[S2]文本\n[S1]文本...",
    ),
    # Generate button
    generate_btn_label=dict(
        en="Generate Audio",
        zh="合成",
    ),
    # Generated audio
    generated_audio_label=dict(
        en="Generated Audio",
        zh="合成的音频",
    ),
    # Examples label
    examples_label=dict(
        en="Podcast Template Examples (Click to Load)",
        zh="播客模板示例 (点击加载)",
    ),
    # Examples generated audio
    examples_generated_audio_label=dict(
        en="Example Audio",
        zh="示例音频",
    ),
    # Warining1: invalid text for prompt
    warn_invalid_spk1_prompt_text=dict(
        en='Invalid speaker 1 prompt text, should not be empty and strictly follow: "xxx"',
        zh='说话人 1 参考文本不合规，不能为空，格式："xxx"',
    ),
    warn_invalid_spk2_prompt_text=dict(
        en='Invalid speaker 2 prompt text, should strictly follow: "[S2]xxx"',
        zh='说话人 2 参考文本不合规，格式："[S2]xxx"',
    ),
    warn_invalid_dialogue_text=dict(
        en='Invalid dialogue input text, should strictly follow: "[S1]xxx[S2]xxx..."',
        zh='对话文本输入不合规，格式："[S1]xxx[S2]xxx..."',
    ),
    # Warining3: incomplete prompt info
    warn_incomplete_prompt=dict(
        en="Please provide prompt audio and text for both speaker 1 and speaker 2",
        zh="请提供说话人 1 与说话人 2 的参考语音与参考文本",
    ),
)


global_lang: Literal["zh", "en"] = "en"

def i18n(key):
    global global_lang
    return _i18n_key2lang_dict[key][global_lang]

def check_monologue_text(text: str, prefix: str = None) -> bool:
    text = text.strip()
    # Check speaker tags
    if prefix is not None and (not text.startswith(prefix)):
        return False
    # Remove prefix
    if prefix is not None:
        text = text.removeprefix(prefix)
    text = text.strip()
    # If empty?
    if len(text) == 0:
        return False
    return True

def check_dialect_prompt_text(text: str, prefix: str = None) -> bool:
    text = text.strip()
    # Check Dialect Prompt prefix tags
    if prefix is not None and (not text.startswith(prefix)):
        return False
    text = text.strip()
    # If empty?
    if len(text) == 0:
        return False
    return True

def check_dialogue_text(text_list: List[str]) -> bool:
    if len(text_list) == 0:
        return False
    for text in text_list:
        if not (
            check_monologue_text(text, "[S1]")
            or check_monologue_text(text, "[S2]")
            or check_monologue_text(text, "[S3]")
            or check_monologue_text(text, "[S4]")
        ):
            return False
    return True

def process_single(target_text_list, prompt_wav_list, prompt_text_list, use_dialect_prompt, dialect_prompt_text):
    spks, texts = [], []
    for target_text in target_text_list:
        pattern = r'(\[S[1-9]\])(.+)'
        match = re.match(pattern, target_text)
        text, spk = match.group(2), int(match.group(1)[2])-1
        spks.append(spk)
        texts.append(text)
    
    global dataset
    dataitem = {"key": "001", "prompt_text": prompt_text_list, "prompt_wav": prompt_wav_list, 
             "text": texts, "spk": spks, }
    if use_dialect_prompt:
        dataitem.update({
            "dialect_prompt_text": dialect_prompt_text
        })
    dataset.update_datasource(
        [
           dataitem 
        ]
    )        

    # assert one data only;
    data = dataset[0]
    prompt_mels_for_llm, prompt_mels_lens_for_llm = s3tokenizer.padding(data["log_mel"])  # [B, num_mels=128, T]
    spk_emb_for_flow = torch.tensor(data["spk_emb"])
    prompt_mels_for_flow = torch.nn.utils.rnn.pad_sequence(data["mel"], batch_first=True, padding_value=0)  # [B, T', num_mels=80]
    prompt_mels_lens_for_flow = torch.tensor(data['mel_len'])
    text_tokens_for_llm = data["text_tokens"]
    prompt_text_tokens_for_llm = data["prompt_text_tokens"]
    spk_ids = data["spks_list"]
    sampling_params = SamplingParams(use_ras=True,win_size=25,tau_r=0.2)
    infos = [data["info"]]
    processed_data = {
        "prompt_mels_for_llm": prompt_mels_for_llm,
        "prompt_mels_lens_for_llm": prompt_mels_lens_for_llm,
        "prompt_text_tokens_for_llm": prompt_text_tokens_for_llm,
        "text_tokens_for_llm": text_tokens_for_llm,
        "prompt_mels_for_flow_ori": prompt_mels_for_flow,
        "prompt_mels_lens_for_flow": prompt_mels_lens_for_flow,
        "spk_emb_for_flow": spk_emb_for_flow,
        "sampling_params": sampling_params,
        "spk_ids": spk_ids,
        "infos": infos,
        "use_dialect_prompt": use_dialect_prompt,
    }
    if use_dialect_prompt:
        processed_data.update({
            "dialect_prompt_text_tokens_for_llm": data["dialect_prompt_text_tokens"],
            "dialect_prefix": data["dialect_prefix"],
        })
    return processed_data


def dialogue_synthesis_function(
    target_text: str,
    spk1_prompt_text: str | None = "",
    spk1_prompt_audio: str | None = None,
    spk1_dialect_prompt_text: str | None = "",
    spk2_prompt_text: str | None = "",
    spk2_prompt_audio: str | None = None,
    spk2_dialect_prompt_text: str | None = "",
    seed: int = 1988,
):
    
    seed = int(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Check prompt info
    target_text_list: List[str] = re.findall(r"(\[S[0-9]\][^\[\]]*)", target_text)
    target_text_list = [text.strip() for text in target_text_list]
    if not check_dialogue_text(target_text_list):
        gr.Warning(message=i18n("warn_invalid_dialogue_text"))
        return None

    # Go synthesis
    progress_bar = gr.Progress(track_tqdm=True)
    prompt_wav_list = [spk1_prompt_audio, spk2_prompt_audio]
    prompt_text_list = [spk1_prompt_text, spk2_prompt_text] 
    use_dialect_prompt = spk1_dialect_prompt_text.strip()!="" or spk2_dialect_prompt_text.strip()!=""
    dialect_prompt_text_list = [spk1_dialect_prompt_text, spk2_dialect_prompt_text]
    data = process_single(
        target_text_list,
        prompt_wav_list,
        prompt_text_list,
        use_dialect_prompt,
        dialect_prompt_text_list,
    )
    results_dict = model.forward_longform(
        **data
    )
    target_audio = None
    for i in range(len(results_dict['generated_wavs'])):
        if target_audio is None:
            target_audio = results_dict['generated_wavs'][i]
        else:
            target_audio = torch.concat([target_audio, results_dict['generated_wavs'][i]], axis=1)
    return (24000, target_audio.cpu().squeeze(0).numpy())


def update_example_choices(dialect_key: str):

    if dialect_key == "(无)":
        choices = ["(请先选择方言)"]

        return gr.update(choices=choices, value="(无)"), gr.update(choices=choices, value="(无)")
    
    choices = list(DIALECT_PROMPT_DATA.get(dialect_key, {}).keys())

    return gr.update(choices=choices, value="(无)"), gr.update(choices=choices, value="(无)")

def update_prompt_text(dialect_key: str, example_key: str):
    if dialect_key == "(无)" or example_key in ["(无)", "(请先选择方言)"]:
        return gr.update(value="")
    

    full_text = DIALECT_PROMPT_DATA.get(dialect_key, {}).get(example_key, "")
    return gr.update(value=full_text)


def render_interface() -> gr.Blocks:
    with gr.Blocks(title="SoulX-Podcast") as page:

        with gr.Row():
            lang_choice = gr.Radio(
                choices=["English", "中文"],
                value="English",
                label="Display Language/显示语言",
                type="index",
                interactive=True,
                scale=3,
            )
            seed_input = gr.Number(
                label="Seed (种子)",
                value=1988,
                step=1,
                interactive=True,
                scale=1,
            )

        with gr.Row():

            with gr.Column(scale=1):
                with gr.Group(visible=True) as spk1_prompt_group:
                    spk1_prompt_audio = gr.Audio(
                        label=i18n("spk1_prompt_audio_label"),
                        type="filepath",
                        editable=False,
                        interactive=True,
                    )
                    spk1_prompt_text = gr.Textbox(
                        label=i18n("spk1_prompt_text_label"),
                        placeholder=i18n("spk1_prompt_text_placeholder"),
                        lines=3,
                    )
                    spk1_dialect_prompt_text = gr.Textbox(
                        label=i18n("spk1_dialect_prompt_text_label"),
                        placeholder=i18n("spk1_dialect_prompt_text_placeholder"),
                        value="",
                        lines=3,
                        visible=False,
                    )

            with gr.Column(scale=1, visible=True):
                with gr.Group(visible=True) as spk2_prompt_group:
                    spk2_prompt_audio = gr.Audio(
                        label=i18n("spk2_prompt_audio_label"),
                        type="filepath",
                        editable=False,
                        interactive=True,
                    )
                    spk2_prompt_text = gr.Textbox(
                        label=i18n("spk2_prompt_text_label"),
                        placeholder=i18n("spk2_prompt_text_placeholder"),
                        lines=3,
                    )
                    spk2_dialect_prompt_text = gr.Textbox(
                        label=i18n("spk2_dialect_prompt_text_label"),
                        placeholder=i18n("spk2_dialect_prompt_text_placeholder"),
                        value="",
                        lines=3,
                        visible=False,
                    )

            with gr.Column(scale=2):
                with gr.Row():
                    dialogue_text_input = gr.Textbox(
                        label=i18n("dialogue_text_input_label"),
                        placeholder=i18n("dialogue_text_input_placeholder"),
                        lines=18,
                    )

        # Generate button
        with gr.Row():
            generate_btn = gr.Button(
                value=i18n("generate_btn_label"), 
                variant="primary", 
                scale=3,
                size="lg",
            )
        
        # Long output audio
        generate_audio = gr.Audio(
            label=i18n("generated_audio_label"),
            interactive=False,
        )


        # Generated audio component for the examples table column (not rendered separately)
        examples_generated_audio = gr.Audio(
            label=i18n("examples_generated_audio_label"),
            type="filepath",
            interactive=False,  # Read-only but playable
            visible=False,  # Hidden - only used as a column in the table
        )
        
        with gr.Row():
            inputs_for_examples = [
                spk1_prompt_audio,
                spk1_prompt_text,
                spk2_prompt_audio,
                spk2_prompt_text,
                dialogue_text_input,
                examples_generated_audio,  # This creates the column in the table
            ]
            
            gr.Examples(
                examples=EXAMPLES_LIST,
                inputs=inputs_for_examples,
                label=i18n("examples_label"),
                examples_per_page=5,
            )
        
        # Display area for selected example audio below the table
        examples_generated_audio_display = gr.Audio(
            label=i18n("examples_generated_audio_label"),
            type="filepath",
            interactive=False,  # Read-only but playable
            visible=True,
        )
        
        # Function to update the display audio when an example is clicked
        def update_display_audio_from_example(spk1_audio, spk1_text, spk2_audio, spk2_text, dialogue_text, generated_audio):
            # Return the generated_audio value to display it below the table
            return gr.update(value=generated_audio)
        
        # Update the display audio when any example field changes
        for input_component in inputs_for_examples:
            input_component.change(
                fn=update_display_audio_from_example,
                inputs=inputs_for_examples,
                outputs=[examples_generated_audio_display],
            )
        
        with gr.Accordion("方言提示文本 (Dialect Prompt) 选择器", open=False, visible=False):
            gr.Markdown("选择方言后，请分别为 S1 和 S2 选择一个示例。")
            dialect_selector = gr.Dropdown(
                label="选择方言 (Select Dialect)", 
                choices=DIALECT_CHOICES, 
                value="(无)",
                interactive=True
            )
            with gr.Row():
                s1_dialect_example_selector = gr.Dropdown(
                    label="S1 方言示例 (S1 Dialect Example)", 
                    choices=["(请先选择方言)"], 
                    value="(无)",
                    interactive=True,
                    elem_classes="gradio-dropdown" 
                )
                s2_dialect_example_selector = gr.Dropdown(
                    label="S2 方言示例 (S2 Dialect Example)", 
                    choices=["(请先选择方言)"], 
                    value="(无)",
                    interactive=True,
                    elem_classes="gradio-dropdown" 
                )
        
        dialect_selector.change(
            fn=update_example_choices,
            inputs=[dialect_selector],
            outputs=[s1_dialect_example_selector, s2_dialect_example_selector]
        )
        
        s1_dialect_example_selector.change(
            fn=update_prompt_text,
            inputs=[dialect_selector, s1_dialect_example_selector],
            outputs=[spk1_dialect_prompt_text]
        )
        
        s2_dialect_example_selector.change(
            fn=update_prompt_text,
            inputs=[dialect_selector, s2_dialect_example_selector],
            outputs=[spk2_dialect_prompt_text]
        )

        def _change_component_language(lang):
            global global_lang
            global_lang = ["en", "zh"][lang]
            return [
                
                # spk1_prompt_{audio,text,dialect_prompt_text}
                gr.update(label=i18n("spk1_prompt_audio_label")),
                gr.update(
                    label=i18n("spk1_prompt_text_label"),
                    placeholder=i18n("spk1_prompt_text_placeholder"),
                ),
                gr.update(
                    label=i18n("spk1_dialect_prompt_text_label"),
                    placeholder=i18n("spk1_dialect_prompt_text_placeholder"),
                    visible=False,
                ),
                # spk2_prompt_{audio,text}
                gr.update(label=i18n("spk2_prompt_audio_label")),
                gr.update(
                    label=i18n("spk2_prompt_text_label"),
                    placeholder=i18n("spk2_prompt_text_placeholder"),
                ),
                gr.update(
                    label=i18n("spk2_dialect_prompt_text_label"),
                    placeholder=i18n("spk2_dialect_prompt_text_placeholder"),
                    visible=False,
                ),
                # dialogue_text_input
                gr.update(
                    label=i18n("dialogue_text_input_label"),
                    placeholder=i18n("dialogue_text_input_placeholder"),
                ),
                # generate_btn
                gr.update(value=i18n("generate_btn_label")),
                # generate_audio
                gr.update(label=i18n("generated_audio_label")),
            ]

        lang_choice.change(
            fn=_change_component_language,
            inputs=[lang_choice],
            outputs=[
                spk1_prompt_audio,
                spk1_prompt_text,
                spk1_dialect_prompt_text,
                spk2_prompt_audio,
                spk2_prompt_text,
                spk2_dialect_prompt_text,
                dialogue_text_input,
                generate_btn,
                generate_audio,
            ],
        )
        
        generate_btn.click(
            fn=dialogue_synthesis_function,
            inputs=[
                dialogue_text_input,
                spk1_prompt_text,
                spk1_prompt_audio,
                spk1_dialect_prompt_text,
                spk2_prompt_text,
                spk2_prompt_audio,
                spk2_dialect_prompt_text,
                seed_input,
            ],
            outputs=[generate_audio],
        )
    return page


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--model_path',
                        required=True,
                        type=str,
                        help='model path')
    parser.add_argument('--llm_engine',
                        type=str,
                        default="hf",
                        help='model execute engine')
    parser.add_argument('--fp16_flow',
                        action='store_true',
                        help='enable fp16 flow')
    parser.add_argument('--seed',
                        type=int,
                        default=1988,
                        help='random seed for generation')
    parser.add_argument('--port',
                        type=int,
                        default=7860,
                        help='gradio port for web app')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # Initiate model
    hf_config = SoulXPodcastLLMConfig.from_initial_and_json(
            initial_values={"fp16_flow": args.fp16_flow}, 
            json_file=f"{args.model_path}/soulxpodcast_config.json")
    
    llm_engine = args.llm_engine
    if llm_engine == "vllm":
        if not importlib.util.find_spec("vllm"):
            llm_engine = "hf"
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [WARNING]: No install VLLM, switch to hf engine.")
    config = Config(model=args.model_path, enforce_eager=True, llm_engine=llm_engine,
                    hf_config=hf_config)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    initiate_model(config)
    print("[INFO] SoulX-Podcast loaded")    
    page = render_interface()
    page.queue()
    page.launch(share=False, server_name="0.0.0.0", server_port=args.port)
