from codon_onnx.tokenizer import PackedTokenizer
from codon_onnx.base import CausalLMOnnx
from codon_onnx.service import ModelCard, Service

tokenizer = PackedTokenizer('./motif.vocab')
model = CausalLMOnnx('motifa1.onnx', force_cpu=True)

Service([
    ModelCard(
        model=model,
        tokenizer=tokenizer,
        model_id='Motif-A1',
        owned='CodonProject'
    )
]).run()