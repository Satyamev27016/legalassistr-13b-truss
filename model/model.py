from threading import Thread
from typing import Dict, List

import torch
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    GenerationConfig,
    TextStreamer
)

DEFAULT_SYSTEM_PROMPT = """
 You are Law Sahayak, A Top Quality Lawyer in Indian Law.
    Your skills are :
    1. Drafting Skills: Proficient in drafting various legal documents, including contracts, agreements, petitions, and legal opinions, ensuring precision and adherence to legal norms.
    2. Legal Writing: Exceptional writing skills to convey complex legal concepts in a clear and concise manner, making legal documents easily understandable for clients and other stakeholders.
    3. Document Review: Expertise in systematically reviewing and analyzing legal documents to identify key issues, potential risks, and areas for improvement.
    4. Compliance Knowledge: In-depth understanding of regulatory frameworks and compliance requirements relevant to specific industries or legal matters, ensuring that all documents adhere to applicable laws and regulations.
    5. Legal Editing: Meticulous attention to detail in editing legal documents to eliminate errors, improve clarity, and maintain consistency in language and formatting.
    6. Summarization Skills: Ability to distill complex legal information into concise summaries, facilitating better understanding for clients, colleagues, or other non-legal professionals involved in a case.
    7. Legal Expertise: Possesses strong knowledge of constitutional law and can interpret complex legal principles.
    8. Research and Analysis: Excels in conducting thorough legal research and in-depth case analysis.
    9. Communication and Negotiation: Demonstrates excellent communication and successful negotiation skills.
    However, it's essential to underline the importance of providing accurate information. Rather than inventing or fabricating data, it is more respectable and credible to admit "I don't know" if you are unsure about the correct information.
"""
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None
        self._tokenizer = None

    def load(self):
        self._model = LlamaForCausalLM.from_pretrained(
            "TheBloke/LLaMa-13B-GGML",
            use_auth_token=self._secrets["hf_access_token"],
            device_map="auto",
        )
        self._tokenizer = LlamaTokenizer.from_pretrained(
            "TheBloke/LLaMa-13B-GGML",
            use_auth_token=self._secrets["hf_access_token"]
        )

    def preprocess(self, request: Dict):
        return request
    
    def forward(self, prompt, stream, temp, top_p, top_k, num_beams=1, max_length=2048, **kwargs):
        generation_config = GenerationConfig(
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=1.2,
            max_length=max_length,
            **kwargs
        )

        prompt_wrapped = (
            f"{B_INST} {B_SYS} {DEFAULT_SYSTEM_PROMPT} {E_SYS} {prompt} {E_INST}"
        )

        inputs = self._tokenizer(
            prompt_wrapped,
            return_tensors="pt",
            truncation=True,
            padding=False,
            max_length=2048,
        )

        input_ids = inputs["input_ids"].to("cuda")

        if not stream:
            with torch.no_grad():
                generation_output = self._model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=max_length,
                    early_stopping=True, 
                )

            decoded_output = []
            for beam in generation_output.sequences:
                decoded_output.append(
                    self._tokenizer.decode(beam, skip_special_tokens=True).replace(
                        prompt_wrapped, ""
                    )
                )

            return decoded_output
        streamer = TextStreamer(self._tokenizer),
    
        generation_kwargs = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "streamer": streamer,
        }

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        def inner():
            first = True
            for text in streamer:
                if first:
                    first = False
                    continue
                yield text
            thread.join()

        return inner()

    def predict(self, request: Dict):
        prompt = request.pop('prompt')
        stream = request.pop('stream')
        return self.forward(prompt, stream, **request)
