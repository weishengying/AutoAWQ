from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = '/mnt/infra/weishengying/model/skywork_mixtral'
quant_path = './skywork_mixtral-awq'
quant_config = { "zero_point": False, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config, calib_data='wikitext', export_compatible=True)

# Save quantized model
# model.save_quantized(quant_path)
model.model.save_pretrained(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')