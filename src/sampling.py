from dataclasses import dataclass

@dataclass
class SamplingParams:
    temprature: float  = 1.0
    max_tokens: int = 4096
    ignore_eos: bool = False