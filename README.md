# Using Prompts to Predict ChatGPT Hallucinations with BERT
This is my final project for [MIT's "Intro to Deep Learning" course](https://introtodeeplearning.com/2025/index.html) on open courseware! 

## Overview
I attempted to fine-tune a BERT-base model to predict whether ChatGPT's response to a given prompt will include a hallucination. Ultimately, I didn't achieve an F1 score above 0.5 (F1 is the harmonic mean of precision and recall; the dataset's imbalance means 0.5 is better than random guessing but still indicates poor performance). These findings indicate that detectable patterns in prompts alone may not be sufficient to predict if a hallucination will occur. 

## Objective
If it were possible to predict the likelihood of a hallucination before the LLM responds, additional hallucination prevention measures could be put in place such as applying chain-of-thought prompting or, after response generation, stricter fact-checking. The goal of this experiment was to see if a BERT model might be able to accomplish this task for prompts sent to ChatGPT. 

## Methodology
### Dataset Creation
To create my dataset, I combined segments of the [HaluEval dataset](https://github.com/RUCAIBox/HaluEval), datasets from [LibreEval](https://github.com/Arize-ai/LibreEval), and data from the [ANAH dataset](https://huggingface.co/datasets/opencompass/anah). I manipulated and combined these datasets using Google Colab in a file titled [Colab_for_Generating_Dataset_of_Prompts_with_Binary_Hallucination_Labels.ipynb](https://github.com/108charlotte/ChatGPT-Hallucination-Likelihood-Detection-from-Prompts-with-BERT/blob/main/Colab_for_Generating_Dataset_of_Prompts_with_Binary_Hallucination_Labels.ipynb), which I annotated and added to this repository. This dataset contains 9250 data points, and 75.73% (7005 samples) of prompts in this dataset did not result in a ChatGPT hallucination while 24.27% (2245 samples) did. If you would like to run the Colab on your own device, I also included a [folder](https://github.com/108charlotte/ChatGPT-Hallucination-Likelihood-Detection-from-Prompts-with-BERT/tree/main/Files%20for%20Dataset%20Creation) which contains all necessary files of the original data. However, if you only want to run the Colab for BERT setup and training, I would recommend initially downloading the [final dataset](https://github.com/108charlotte/ChatGPT-Hallucination-Likelihood-Detection-from-Prompts-with-BERT/blob/main/final_dataset.csv) instead, which is the result of running the dataset generation Colab. 

### Model Architecture
I used the [BERT base uncased](https://huggingface.co/google-bert/bert-base-uncased) model because it can detect patterns in language, which may help it detect patterns in prompts likely to cause hallucinations, and is already trained on a large data corpus, so I won't have to supply enough data to provide an understanding of language. The uncased variation allows for easier training with a smaller dataset since the model doesn't need to learn patterns in capitalization, and the base model allows for better generalization with limited data than the large model. I defined my model with BertForSequenceClassification. 

### Methods for Lessening Impact of Class Imbalance
In order to discourage the model from simply predicting the majority class (no hallucination, or 0), I incorporated a custom weighted trainer, with the Pytorch implementation written by Claude, and added an additional weighting to the positive class (I experimented with what this value would be in the hyperparameter search). 

### Hyperparameter Search
After defining my model, I completed a hyperparameter search across different LoRA values and pos_weight values. I used Claude to write the search so that I could focus my efforts on analyzing the model's performance and improving it. I only used one learning rate since I needed the search to complete within 3 hours. I experimented with all combinations of the following hyperparameter values: 8 and 16 for the r value in LoRA, 16 and 32 the alpha value in LoRA, 0.15 and 0.2 for the dropout value in LoRA, 1.2 and 1.3 for the positive weight multiplier (in the custom weighted loss function). Each search permutation used the same learning rate (2e-4) since testing multiple along with testing many other hyperparameter values would be too computationally expensive. I chose this learning rate, which is higher than typically recommended for BERT, due to needing to see model improvement in a short number of epochs. 


## Results and Analysis
While looking at models, I rated them using their F1 scores since this model's goal is to predict as many true positives as possible without flagging too many false ones. 
The highest maximum F1 value achieved by a configuration was 0.4678 (configuration 15, epoch 5, r=16, lora_alpha=32, lora_dropout=0.2, pos_weight_mult=1.2), and the lowest maximum F1 value for a configuration was 0.4436 (configuration 3, epoch 1, r=8, lora_alpha=16, lora_dropout=0.2, pos_weight_mult=1.2). These values are better than if the model had simply predicted all positives (F1=0.3906), and better than if it had guessed randomly (F1=0.3268), but since no combination was able to achieve an F1 score at or above 0.5, the model still has significant room for improvement in reliably predicting hallucinations. 
Additionally, the fact that all tested hyperparameter combinations had F1 scores in a similar range suggests that hyperparameter tuning is unlikely to significantly boost the model's performance. 

### Confusion Matrix
![Confusion matrix for the best model](https://hc-cdn.hel1.your-objectstorage.com/s/v3/6e19d99aeadebafd_confusion_matrix_best.png)
*Best model (Config 15, Epoch 5): 167 true positives, 323 false positives, 57 false negatives, and 378 true negatives*

![Confusion matrix for the worst model](https://hc-cdn.hel1.your-objectstorage.com/s/v3/67d5032d98ca27c0_confusion_matrix_worst.png)
*Worst model (Config 3, Epoch 1): 179 true positives, 404 false positives, 45 false negatives, and 297 true negatives*

Both models have a high rate of false positives, with approximately one in every three positive predictions being a true positive and two being false alarms. However, both models have high recall (Config 3 recall in epoch 1 is 80%, Config 15 recall in epoch 5 is 75%). While high recall allows the model to detect more true positives, it is at the cost of precision which decreases the model's F1 and creates many false alarms. 

### Training v. Validation Loss
![Plot of Training and Validation Loss for Best Model](https://hc-cdn.hel1.your-objectstorage.com/s/v3/202bcc3a870081b5_loss_best.png)
*Best model (Config 15, Epoch 5): Overfitting begins after Epoch 3*

![Plot of Training and Validation Loss for Worst Model](https://hc-cdn.hel1.your-objectstorage.com/s/v3/272b23ab65cee90b_loss_worst.png)
*Worst model (Config 3, Epoch 1): Overfitting begins immediately before Epoch 4*

The best model's training and validation loss plot indicates overfitting beginning after Epoch 3, and the worst model at Epoch 4. In the end, both the best and worst models ended up with similar loss trajectories. These results suggest that the dataset may have been too small to train a BERT model on or, since recall is known to be high, that the model isn't detecting any reliable characteristics in true positive prompts which would allow it to have a high precision. 


## Takeaways and Future Work
### Limitations
My largest limitation was computational cost and dataset size. In the future, it would be helpful to be able to test smaller learning rates since, under time constraints, I needed to see results in a small number of epochs. Regarding my dataset, I was only able to create a dataset of a little over 9k samples which is on the lower end for BERT fine-tuning. 

### Conclusion
Although F1 values above those from random guessing indicate that the model successfully learned some patterns in the data, the fact that the F1 values are still below 0.5 for all hyperparameters reveals that the model isn't learning patterns in prompts which can reliably predict hallucinations. The model's high recall can be attributed to the positive weight multiplier which encourages the model to aggressively predict positive labels. LoRA parameter tuning didn't help because there aren't strong enough patterns to detect in prompts which correspond with ChatGPT hallucinations. 

### Future Work
#### Minor Tweaks
In the future, it would be beneficial to run this experiment with a lower learning rate, which may allow the model to detect more subtle patterns. Additionally, curating a larger dataset of 12-15k+ samples may help prevent overfitting.

#### New Directions
However, since this model struggled detecting patterns in prompts alone, it could be beneficial to explore how prompt metadata, such as question type or tone, may influence the likelihood of a hallucination. Additionally, understanding which patterns this model is picking up could reveal more targeted directions for future exploration, for example if emotionally charged phrases or contested past events are leading to a higher hallucination rate. 
