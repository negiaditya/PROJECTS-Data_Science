from datasets import load_metric
from transformers import TrainingArguments, Trainer
from transformers import LayoutLMv3ForTokenClassification,AutoProcessor
from transformers.data.data_collator import default_data_collator
import torch
from datasets import load_from_disk
import warnings
warnings.filterwarnings('ignore')
from config import trainingParams, preprocessed_path, save_model_path

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    results = metric.compute(predictions=true_predictions, references=true_labels,zero_division='0')
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

if __name__ == "__main__":
	input_path = preprocessed_path
	train_dataset = load_from_disk(f'{input_path}train_split')
	eval_dataset = load_from_disk(f'{input_path}eval_split')
	print(train_dataset)

	label_list = train_dataset.features["labels"].feature.names
	num_labels = len(label_list)
	label2id, id2label = dict(), dict()
	for i, label in enumerate(label_list):
	    label2id[label] = i
	    id2label[i] = label

	metric = load_metric("seqeval")
	return_entity_level_metrics = False

	model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base",id2label=id2label,label2id=label2id)
	processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

	training_args = TrainingArguments(output_dir='test',
                                  # max_steps=trainingParams['MAX_STEPS'],
                                  num_train_epochs=trainingParams['NUM_TRAIN_EPOCHS'],
                                  logging_strategy="epoch",
                                  save_total_limit=1,
                                  per_device_train_batch_size=trainingParams['PER_DEVICE_TRAIN_BATCH_SIZE'],
                                  per_device_eval_batch_size=trainingParams['PER_DEVICE_EVAL_BATCH_SIZE'],
                                  learning_rate=trainingParams['LEARNING_RATE'],
                                  evaluation_strategy="epoch",
                                  save_strategy="epoch",
                                  # eval_steps=trainingParams['EVAL_STEPS'],
                                  load_best_model_at_end=True,
                                  metric_for_best_model="f1")

	# Initialize our Trainer
	trainer = Trainer(
	    model=model,
	    args=training_args,
	    train_dataset=train_dataset,
	    eval_dataset=eval_dataset,
	    tokenizer=processor,
	    data_collator=default_data_collator,
	    compute_metrics=compute_metrics,
	)

	print("Training started")
	trainer.train()

	print("Evaluation")
	trainer.evaluate()

	print("Saving trained layoutML model")
	OUTPUT_DIR = save_model_path
	#model.save_pretrained(OUTPUT_DIR)
	torch.save(model,f'{OUTPUT_DIR}')