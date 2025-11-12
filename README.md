# full-stackRENAL
# AI empowers full-stack smart diagnosis
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/Greenlovecola8023/full-stackRENAL)

This project developed a full-stack diagnostic model for renal cell carcinoma, including four tasks: tissue detection, subtype prediction, auto-grading and prognostication. It is completed based on the features extracted by Prov-GigaPath combined with the abmil architecture. All required packages are listed in `requirements.txt`.
![overview.png](https://github.com/Green8023/full-stackRENAL/blob/main/overview.png)
## preprocessing
First, cut the slide into patches. The process of splitting can be referred to ```preprocess.py```. Then, Prov-GigaPath is used to extract features from the cut patches. For details on Prov-GigaPath please refer to their official repository:https://github.com/prov-gigapath/prov-gigapath

## tissue detection task
For the task of tissue detection, which focused on identifying five distinct tissue components (normal tissue, tumor tissue, pseudocapsule, necrosis, and sarcomatoid differentiation), each WSI was manually annotated for these five regions by four experienced pathologists, and model performance was primarily evaluated based on patch-level predictive accuracy. 
Please see the code ```task_tissue_detection.py```

## subtype prediction
For the pathological subtype classification of renal cell carcinoma (RCC), only RCC subtypes with a sample size exceeding 10 patients were included to ensure statistical robustness. Model performance was primarily evaluated using slide-level predictive accuracy. A total of nine pathological subtypes were analyzed for classification.
Please see the code ```task_pathocls_prediction.py```

## tumor grading
For the nuclear grading task, we filtered tumor-related patches and grouped them into bags by patient ID. The original four ISUP grades were simplified into a binary task, grades 1 and 2 were grouped as low grade, while grades 3 and 4 were grouped as high grade.
Please see the code ```task_nuclear_grading.py```

## prognositication
For prognosis prediction, training was conducted with patients who had follow-up information. The patients were divided into a high-risk group and a low-risk group, and the probability of high risk was given by the model after learning as the prognosis score.
Please see the code ```task_pathocls_prediction.py```

## visulization
Some visulization tools are provided. For tissue detection and subtype prediction, we make it visible and were able to mark it in different colors after model inference.
Please see the code in visulization folder.

The pre-trained model weights are public in huggingface. https://huggingface.co/Greenlovecola8023/full-stackRENAL
