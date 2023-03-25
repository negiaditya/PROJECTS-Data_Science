STEPS:
1. Given input images have been split into train-test and saved under inv_train and inv_test folder respectively.
2. We annotate the images in inv_train folder using UIBAI annotator tool and save the annotation in OCR Processed format under layoutLM_input_format folder. It contains train images, all.txt all_box.txt, all_image.txt and all_labels.txt files.
3. We have layoutLM_finetune_and_inference.ipynb file which containes code for preprocessing, training and inference done in google colab. Look there for results.
4. We get train_split & eval_split under processed_data folder after preprocessing the input layoutLM_finetune_and_inference folder.
5. After training is completed we get the saved model in '.pth' format in model folder.
6. Inference is done using inv_test,model folders as inputs and we save the json output and images with result bounding box in out folder.
------------------------------------------------------------------------------------------------------------------------------------------

Creating pipelines/scripts for the above mentioned processes.

Training:
1.We have created the config.py with all parameters and paths.
2.We provide the annotation_path which is the input folder used for preprocessing the data. Run preprocessor.py which also takes input the output preprocessed data path split into train,eval using the train_split_ratio variable.
3.We will have the postprocessed data in processed_data folder, we will now run train.py which takes trainingParams and processed_path as input and after training saves the model in save_model_path as .pth extension.

Inference:
4.Finally we can run a batch of images saved in a folder and pass it to inference.py using test_dir variable, output will contain images with bounding boxes and outputJson stored in test_output_path. We load the model using save_model_path. Make sure to run GPU.
5. We have saved weights of the model as well and that can be used in inference as well. But it also came above 400mb.
6. Please download the model from here 'https://drive.google.com/file/d/1VDCGA-FjMTUympHWUw9F3sPfYn5237tH/view?usp=sharing' and and save it in ./model folder.
7. Due to size limit of 50mb pretrained model can be downloaded and or after running training script.
-------------------------------------------------------------------------------------------------------
In the python environment/terminal run the following:
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
pip install pytesseract
python -m pip install -q 'git+https://github.com/facebookresearch/detectron2.git'
pip install -q git+https://github.com/huggingface/transformers.git
pip install -q git+https://github.com/huggingface/datasets.git "dill<0.3.5" seqeval

# Change the config file which has relative paths to given folder.
# For training pipeline run:
python preprocessor.py
python train.py

# For inference pipeline run: toggle GPU
python inference.py
----------------------------------------------------------------------------------------------------------
Look at run_pipeline.ipynb for the running of scripts.