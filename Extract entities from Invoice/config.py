
annotation_path = './layoutLM_input_format/'
preprocessed_path = './processed_data/'
train_test_split_ratio = 0.1
save_model_path = './model/layoutlmv3_final.pth'
test_dir = './inv_test/'
test_output_path = './inference_output/'
trainingParams = {	'NUM_TRAIN_EPOCHS': 150,
					'MAX_STEPS': 1500,
					'PER_DEVICE_TRAIN_BATCH_SIZE': 1,
					'PER_DEVICE_EVAL_BATCH_SIZE': 1,
					'LEARNING_RATE': 4e-5,
					'EVAL_STEPS': 100 }