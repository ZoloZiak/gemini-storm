import os, sys, re, json, time, types, subprocess, requests
import numpy as np

# This script is a bridge between Stanford STORM and Gemini CLI.
# It allows STORM to use the local Gemini CLI as its LLM engine, 
# effectively bypassing API key/auth issues by using the CLI's existing session.

# --- MOCKS & MONKEYPATCHING ---
class MockLitellm:
    def __init__(self):
        self.drop_params = True
        self.telemetry = False
        self.cache = None
    def embedding(self, *args, **kwargs):
        return {"data": [{"embedding": [0.0]*384}], "usage": {"total_tokens": 0}}
    def completion(self, *args, **kwargs):
        raise Exception("litellm.completion called unexpectedly")

sys.modules['litellm'] = MockLitellm()
sys.modules["litellm.caching"] = types.ModuleType("litellm.caching")
caching_inner_mock = types.ModuleType("litellm.caching.caching")
caching_inner_mock.Cache = type("Cache", (), {"__init__": lambda *a, **k: None})
sys.modules["litellm.caching.caching"] = caching_inner_mock

import knowledge_storm.encoder
from sentence_transformers import SentenceTransformer

class LocalEncoder:
    def __init__(self, **kwargs):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
        self.total_token_usage = 0
    def encode(self, texts):
        if isinstance(texts, str): texts = [texts]
        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings)
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings.astype(np.float32)
    def get_total_token_usage(self, reset=False): return 0

knowledge_storm.encoder.Encoder = LocalEncoder

from knowledge_storm.storm_wiki.engine import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
import dspy

# --- NESTED GEMINI WRAPPER ---
class VertexCompatibleModel(dspy.dsp.LM):
    def __init__(self, model_name="gemini-2.0-flash"):
        super().__init__(model=model_name)
        self.model_name = model_name
        self.kwargs = {"max_output_tokens": 8000, "temperature": 0.0}

    def basic_request(self, prompt, **kwargs):
        return self.__call__(prompt, **kwargs)

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        try:
            # Assumes gemini is in PATH. Force YOLO mode to avoid approval prompts.
            gemini_path = "gemini" 
            
            head_instruction = "Respond ONLY with the completion. No intro/outro. PROMPT: "
            
            process = subprocess.Popen(
                [gemini_path, "--approval-mode", "yolo", "-p", "-", "--raw-output", "--accept-raw-output-risk"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                shell=True
            )
            
            full_prompt = head_instruction + prompt
            stdout, stderr = process.communicate(input=full_prompt, timeout=900)

            if process.returncode != 0:
                # Filter out the 'True color' warning which is not a real error
                if "True color" in stderr and not stdout.strip():
                    raise Exception(f"Gemini CLI Error: {stderr}")
                elif "True color" in stderr:
                    pass # Ignore warning if we have output
                else:
                    raise Exception(f"Gemini CLI Error: {stderr}")
            
            # Clean output from common CLI artifacts
            clean_output = stdout.replace("Warning: True color (24-bit) support not detected.", "").strip()
            return [clean_output]
            
        except Exception as e:
            raise Exception(f"Gemini CLI Bridge Error: {str(e)}")

class UltraHybridRM(dspy.Retrieve):
    def __init__(self, you_key, k=10):
        super().__init__(k=k)
        self.you_key = you_key
    def forward(self, query_or_queries, exclude_urls=[]):
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        results = []
        for q in queries:
            try:
                y_res = requests.get('https://api.you.com/v1/search', headers={'X-API-Key': self.you_key}, params={'query': q}, timeout=15).json()
                for r in y_res.get('hits', []):
                    snippet = r.get('snippet') or r.get('description') or 'No content available'
                    results.append({'description': snippet[:300], 'snippets': [snippet], 'title': r.get('title', 'Untitled'), 'url': r.get('url', 'http://none')})
            except: pass
        if not results: results = [{'description': 'None', 'snippets': ['None'], 'title': 'None', 'url': 'http://none'}]
        return results[:self.k]

def run_storm(topic, output_dir):
    you_key = os.environ.get('YOU_API_KEY')
    if not you_key:
        print("Error: Missing YOU_API_KEY in environment.")
        sys.exit(1)
        
    runner_args = STORMWikiRunnerArguments(output_dir=output_dir, max_conv_turn=3, max_perspective=3, search_top_k=5)
    rm = UltraHybridRM(you_key=you_key)
    lm = VertexCompatibleModel()
    lm_configs = STORMWikiLMConfigs()
    lm_configs.set_conv_simulator_lm(lm)
    lm_configs.set_question_asker_lm(lm)
    lm_configs.set_outline_gen_lm(lm)
    lm_configs.set_article_gen_lm(lm)
    lm_configs.set_article_polish_lm(lm)
    runner = STORMWikiRunner(runner_args, lm_configs, rm)
    runner.run(topic=topic, do_research=True, do_generate_outline=True, do_generate_article=True, do_polish_article=True)

if __name__ == '__main__':
    if len(sys.argv) > 1: target_topic = sys.argv[1]
    else: target_topic = 'Gemini CLI Extensions'
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', target_topic)[:50]
    out_dir = f'./research_{safe_name}_{int(time.time())}'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    run_storm(target_topic, out_dir)
