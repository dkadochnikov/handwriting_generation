import os
import shutil
import math

from generate import generate_conditionally
from gpt_finetune import generate_response
from utilz import plot_stroke, plot_concat

from transformers import GPT2LMHeadModel, GPT2Tokenizer


model_name = "fine_tuned_gpt2_shakespeare"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

prompt_text = "THE PROLOGUE."
responses = generate_response(prompt_text, model, tokenizer, 500)

for response in responses:
    directory = "plots"

    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    print(response)
    line_stroke_num = 0
    for line in response.split("\n"):
        for i in range(0, math.ceil(len(line)/50) * 50, 50):
            line_stroke_num += 1
            line_slice = line[i:i+50]
            line_strokes = []
            for word in line_slice.split():
                if word:
                    stroke = generate_conditionally(word.lower())
                    line_strokes.append(stroke)
            if line_strokes:
                plot_stroke(line_strokes, line_stroke_num)

plot_concat()
