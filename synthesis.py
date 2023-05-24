from generate import generate_conditionally
from text_generation import generate_text


input_text = generate_text("Once upon a time ")
words = input_text.split()
for i in range(0, len(words) - 1, 2):
    line = words[i] + " " + words[i+1]
    print(line)
    generate_conditionally(line)
