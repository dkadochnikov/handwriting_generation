from generate import generate_conditionally
from gpt_finetune import generate_response

from transformers import GPT2LMHeadModel, GPT2Tokenizer


model_name = "fine_tuned_gpt2_shakespeare"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt_text = "THE PROLOGUE."
responses = generate_response(prompt_text, model, tokenizer, 500)

for response in responses:
    print(response)
    words = response.split()
    for i in range(0, len(words) - 1, 2):
        line = words[i] + " " + words[i + 1]
        generate_conditionally(line)
