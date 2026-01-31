from src.config import Config



class LLMEngine:
    def __init__(
        self,
        model_path: str,
    ):
    config = Config(model_path)
    self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    Sequence.block_size = config.kvcache_block_size
    self.model_runner = ModelRunner(config)
    self.scheduler = Scheduler(config)

    def step(self) -> list[tuple[int, list[int]]]:
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finised()]
        return outputs

    def generate(
        self,
        prompts: list[list[int] | str],
        sampling_params: SamplingParams | list[SamplingParams]
    ) -> list[dict]:
    prompt_token_ids = []
    for prompt in prompts:
        if isinstance(prompt, str):
            prompt_token_ids.append(self.tokenizer.encode(prompt))
        else:
            prompt_token_ids.append(prompt)

    if isinstance(sampling_params, SamplingParams):
        sampling_params_list = [sampling_params] * len(prompts)
    else:
        sampling_params_list = sampling_params
        assert len(sampling_params_list) == len(prompts), \
            f"the length of sampling_params_list ({len(sampling_params_list)}) must be equal to the length of prompts ({len(prompts)})"

    outputs = {}
    for prompt, sampling_params in zip(prompt_token_ids, sampling_params_list):
        self.scheduler.add_request(prompt, sampling_params)
    while not self.scheduler.is_finished():
        output = self.step()
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids
    outputs = [outputs[i] for i in sorted(outputs.keys())]
    outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
    return outputs