import os
import requests
from openai import AzureOpenAI, OpenAI
import time

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class LLM:
    def __init__(self, model='chatgpt', config=None, use_vllm: bool = False):
        print(f'using model: {model}')
        self.use_vllm = getattr(config, "get", None) and config.get('model.use_vllm', default=False) if config is not None else use_vllm
        self.model_choice = model
        self.input_tokens = 0
        self.output_tokens = 0
        self.config = config
        self.t = self.config.get('model.temperature', default=None) if self.config is not None else None
        self.max_new_tokens = (
            self.config.get('model.max_new_tokens', default=128) if self.config is not None else 128
        )

        local_qwen_models = [
            'qwen2.5-0.5b-instruct',
            'qwen2.5-3b-instruct',
            'qwen2.5-7b-instruct',
            'Qwen3-32B',
            'DeepSeek-R1-0528-Qwen3-8B',
            'Qwen3-8B',
            'merged-sft-qwen2.5-7b-instruct',
            'customdata-sft-qwen2.5-7b-instruct',
            'customdata-sft-qwen2.5-7b-instruct-v2',
        ]

        if ',' in model and model not in local_qwen_models:
            self.model_choice = model.split(',')[1]
            self.chat = self.proxy_chat
        else:
            if model in local_qwen_models:
                self.model, self.tokenizer = self._init_qwen(model)
                self.chat = self.qwen_chat
            else:
                self.model = self._init_model(model)
                self.chat = self._init_chat(model)

        print('model choice:', self.model_choice)

    def proxy_chat(self, content):
        base_url = "<TO_BE_FILLED>"
        api_key = os.getenv("MY_API_KEY")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"  
        }
        
        if self.t is not None:
            data = {
                "model": self.model_choice,
                "messages": [
                    {"role": "user", "content": content}
                ],
                "temperature": self.t
            }
        else:
            data = {
                "model": self.model_choice,
                "messages": [
                    {"role": "user", "content": content}
                ],
            }
        while True:
            try:
                response = requests.post(base_url, headers=headers, json=data)
                response.raise_for_status()  
                if response.status_code != 200:
                    print(f"Request failed with status code {response.status_code}")
                    print("Response headers:", response.headers)
                    try:
                        print("Response JSON:", response.json())
                    except Exception:
                        print("Response text:", response.text)
                else:
                    break
            except Exception as e:
                print(f'Exception {e},retry in 20s')
                time.sleep(20)
        response = response.json()
        if 'usage' in response:
            self.input_tokens += response['usage'].get('prompt_tokens', 0)
            self.output_tokens += response['usage'].get('completion_tokens', 0)
        return response['choices'][0]['message']['content']

    def _init_chat(self, model):
        if model == 'chatgpt':
            return self.gpt_chat
        elif model == 'llama':
            return self.llama_chat
        elif model == 'gemini':
            return self.gemini_chat
        elif model == 'deepseek':
            return self.deepseek_chat

    def _init_model(self, model):
        if model == 'chatgpt':
            return self._init_chatgpt()
        elif model == 'llama':
            return self._init_llama()
        elif model == 'gemini':
            return self._init_gemini()
        elif model == 'deepseek':
            return self._init_deepseek()
        elif model in [
            'qwen2.5-0.5b-instruct',
            'qwen2.5-3b-instruct',
            'qwen2.5-7b-instruct',
            'Qwen3-32B',
            'DeepSeek-R1-0528-Qwen3-8B',
            'Qwen3-8B',
            'merged-sft-qwen2.5-7b-instruct',
            'customdata-sft-qwen2.5-7b-instruct',
            'customdata-sft-qwen2.5-7b-instruct-v2',
        ]:
            return self._init_qwen(model)

    def _init_deepseek(self):
        client = OpenAI(api_key="<TO_BE_FILLED>", base_url="https://api.deepseek.com")
        return client
    
    def deepseek_chat(self, content):
        response = self.model.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful chemist and biologist"},
                {"role": "user", "content": content},
            ],
            stream=False
        )
        return response.choices[0].message.content
    
    def _init_gemini(self):
        print("Gemini model is not configured in this environment.")
        return None
    
    def gemini_chat(self, content):
        if self.model is None:
            raise RuntimeError("Gemini model is not initialized.")
        response = self.model.generate_content(content)
        print(response.text)
        return response.text

    def _init_llama(self):
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )
        return client
    
    def llama_chat(self, content):
        completion = self.model.chat.completions.create(
            model="NousResearch/Meta-Llama-3-8B-Instruct",
            messages=[
                {"role": "user", "content": content}
            ]
        )
        return completion.choices[0].message.content
        


    def _init_chatgpt(self):
        resource_name = "<TO_BE_FILLED>"
        endpoint = f"https://{resource_name}.openai.azure.com/"
        api_version = "2024-02-15-preview"

        azure_credential = ChainedTokenCredential(
            AzureCliCredential(),
            DefaultAzureCredential(
                exclude_cli_credential=True,
                exclude_environment_credential=True,
                exclude_shared_token_cache_credential=True,
                exclude_developer_cli_credential=True,
                exclude_powershell_credential=True,
                exclude_interactive_browser_credential=True,
                exclude_visual_studio_code_credentials=True,
                managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
            )
        )

        token_provider = get_bearer_token_provider(azure_credential,
            "https://cognitiveservices.azure.com/.default")
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider
        )
        
        
        return client
    
    def gpt_chat(self, content):
        completion = self.model.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who can propose novel and powerful molecules based on your domain knowledge."},
                {
                    "role": "user",
                    "content": content,
                },
            ],
        )
        res = completion.choices[0].message.content
        return res

    def _init_qwen(self, model='qwen2.5-3b-instruct'):
        if model == 'qwen2.5-0.5b-instruct':
            model_path = "<TO_BE_FILLED>"
        elif model == 'qwen2.5-3b-instruct':
            model_path = "<TO_BE_FILLED>"
        elif model == 'Qwen3-32B':
            model_path = "<TO_BE_FILLED>"
        elif model == 'qwen2.5-7b-instruct':
            model_path = "<TO_BE_FILLED>"
        elif model == 'DeepSeek-R1-0528-Qwen3-8B':
            model_path = "<TO_BE_FILLED>"
        elif model == 'Qwen3-8B':
            model_path = "<TO_BE_FILLED>"
        elif model == 'customdata-sft-qwen2.5-7b-instruct':
            model_path = "<TO_BE_FILLED>"
        else:
            raise ValueError(f"Unknown Qwen model: {model}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_obj = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        if torch.cuda.is_available():
            model_obj = model_obj.cuda()
        return model_obj, tokenizer

    def qwen_chat(self, content: str) -> str:
        if not hasattr(self, 'tokenizer') or not hasattr(self, 'model'):
            raise RuntimeError("Tokenizer or model not initialized for Qwen.")

        messages = [{"role": "user", "content": content}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.tokenizer([text], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "<|im_start|>assistant" in output_text:
            output_text = output_text.split("<|im_start|>assistant")[-1]
        elif "assistant" in output_text:
            output_text = output_text.split("assistant")[-1]
        output_text = output_text.replace("<|im_end|>", "").strip()
        return output_text

    def reset_model(self):
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'base_model'):
            del self.base_model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        torch.cuda.empty_cache()

    def load_model_from_path(self, model_path: str):
        if self.model_choice != 'qwen2.5-7b-instruct':
            print(f"Skipping model loading: `load_model_from_path` was called on '{self.model_choice}', "
                  f"but it's designed only for 'qwen2.5-7b-instruct'.")
            return

        print(f"Loading new model from path: {model_path}")
        self.reset_model()

        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.chat = self.qwen_chat
        print(f"Successfully loaded model weights for: {self.model_choice}")

    def load_lora_weights(self, lora_path: str):
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        if getattr(self, 'model_choice', None) == 'qwen2.5-7b-instruct':
            from peft import PeftModel
            self.reset_model()
            model_path = "<TO_BE_FILLED>"
            self.base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if torch.cuda.is_available():
                self.base_model = self.base_model.cuda()
            print(f"Loading lora weights: {lora_path}")
            self.model = PeftModel.from_pretrained(self.base_model, lora_path)
        else:
            print("Current model does not support dynamic lora weights loading")
