import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments, PreTrainedTokenizer


class GPT2FineTuning:

    def __init__(self) -> None:
        model_name: str = "gpt2"
        path_model: str = 'src/model/gpt2-finetuned'
        if os.path.exists(path_model):
            self.model = AutoModelForCausalLM.from_pretrained(path_model)
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(path_model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)

    def train(self, path: str) -> None:
        dataset_path = path
        dataset = self.load_dataset(dataset_path, self.tokenizer)
        data_collator = self.get_data_collator(self.tokenizer)

        training_args = TrainingArguments(
            output_dir="./gpt2-finetuned",
            overwrite_output_dir=True,
            num_train_epochs=8,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset
        )

        trainer.train()

        self.model.save_pretrained("./gpt2-finetuned")
        self.tokenizer.save_pretrained("./gpt2-finetuned")

    def predict(self, input_text: str, max_length: int = 50) -> str:

        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1)

        output: str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output

    @staticmethod
    def load_dataset(file_path: str, tokenizer: PreTrainedTokenizer, block_size: int = 128) -> TextDataset:
        return TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=block_size
        )

    @staticmethod
    def get_data_collator(tokenizer: PreTrainedTokenizer) -> DataCollatorForLanguageModeling:
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )


if __name__ == "__main__":
    GPT2_model = GPT2FineTuning()
    # GPT2_model.train("../../data/output.txt")
    print(GPT2_model.predict("Hola ", 150))
