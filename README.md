# UoM_SurgicalAI_SurgVU_Challenge2025

## Introduction: 
This is the code repository of the SurgVU25 Phase 2 Team UOM-SurgicalAI.
The repository provides:

Code for building Docker images (MICCAI2025_EndoVQA_Docker)

Fine-tuning scripts (Fine-Tuning-Scripts)

Inference code for zero-shot models and fine-tuned models (Inferencing_Scripts)

The dataset used for training and the corresponding dataset creation code (Dataset_Building)

## Dataset:
As the part of weakly supervised training, we have prepare some data from given training videos and validation clips. Our annotation can be found here: [Google Drive](https://drive.google.com/drive/folders/1wxX2dN7VMr1v-rHypSJXQHDIfrpi9A4r?usp=sharing)

The dataset building details and progress is introduced in the readme file in the 'Dataset_Building' folder

## Pretrained Weights
The pretrained weights will be released soon

## Fine-tuning & Inference required libs
the reuqired libs are listed in the requirements.txt file

## Training Command
Take file DCT_fine_tune.py as example. To run the Fine-tuning code, you need to modify the path of your model downloading path in line 41 and 42 and set your Huggingface token in line 36.
Then you can run the code through command below:
```
python  DCT_fine_tune.py`
    --train_data_path "C:\path\to\your\dataset" `
    --val_data_path "C:\path\to\your\dataset" `
```

## Inference Command
Take file InferenceVL_3_5-4B_Zero-Shot-Inferencing.py as example. You need to set the dataset path in line 568, set the output path in line 589, and set your Huggingface token in line 37.
```
python InferenceVL_3_5-4B_Zero-Shot-Inferencing.py
```

## Docker Building
Please follow the official instruction of SurgVU25 

## Acknowledgement
The implementation of our model relies on resources from <a href = "https://github.com/OpenGVLab/InternVL">InternVL3 model</a>,<a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, <a href="https://github.com/huggingface/peft">PEFT</a>. We thank the original authors for their open-sourcing.
