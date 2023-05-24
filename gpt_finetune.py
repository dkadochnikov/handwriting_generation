from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


def load_dataset(train_file, test_file, tokenizer):
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_file, block_size=128)
    test_dataset = TextDataset(tokenizer=tokenizer, file_path=test_file, block_size=128)
    return train_dataset, test_dataset


def generate_response(prompt_text, model, tokenizer, max_length=50, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    # Generate response
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    responses = []
    for response_id in output_sequences:
        response = tokenizer.decode(response_id, skip_special_tokens=True)
        responses.append(response)

    return responses


def main():
    model_name = "gpt2"
    config = GPT2Config.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name, config=config)

    train_file = "data/shakespeare_test_data.txt"
    test_file = "data/shakespeare_train_data.txt"

    train_dataset, test_dataset = load_dataset(train_file, test_file, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="output",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    trainer.save_model("fine_tuned_gpt2_shakespeare")
    tokenizer.save_pretrained("fine_tuned_gpt2_shakespeare")


if __name__ == "__main__":
    main()
