from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(input_str, model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    inputs = tokenizer.encode(input_str, return_tensors='pt')
    outputs = model.generate(inputs, max_length=500, temperature=0.7, num_return_sequences=1, do_sample=True)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
