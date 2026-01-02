# Predicting ChatGPT Hallucinations with Prompts using BERT
My final project for MIT's open-source Intro to Deep learning course. 

## Objective
The goal of this project is to detect whether or not a prompt is likely to cause ChatGPT (for this dataset, but could also use other datasets for other models) to hallucinate, and if it is to put in place further precautions against hallucinations such as increasing RAG usage or fact verification on results. Rather than applying higher scrutiny to every task, this model would allow certain responses to be selectively analyzed for accuracy, resulting in an overal decrease in the computational cost associated with receiving a response.  

## Process
### Dataset Creation
I combined the [HaluEval general dataset](https://github.com/RUCAIBox/HaluEval) (4500 samples), several of [LibreEval's datasets](https://github.com/Arize-ai/LibreEval) (4200 samples), and the [ANAH dataset](https://huggingface.co/datasets/opencompass/anah) (783 samples) to create a ~9500 sample dataset consisting of two columns: an input (type string, called a prompt) and a binary label for whether or not that prompt resulted in a ChatGPT hallucination (type integer, 0/1). In the final dataset, there are 7291 samples of prompts which did not cause hallucinations and 2245 of prompts which caused hallucinations which makes it imbalanced. The colab link for generating this dataset can be found [here](https://colab.research.google.com/drive/1bDYQKSXsFnlV4rk53V5sWFWzzFaQr-aB#scrollTo=UAl4fhEWaBNL) and on GitHub [here](https://github.com/108charlotte/ChatGPT-Hallucination-Likelihood-Detection-using-Prompts-with-BERT.ipynb/blob/main/Colab_for_Generating_Dataset_of_Prompts_with_Binary_Hallucination_Labels.ipynb). The actual dataset can be downloaded in csv form [here](https://github.com/108charlotte/ChatGPT-Hallucination-Likelihood-Detection-using-Prompts-with-BERT.ipynb/blob/main/final_dataset.csv). In my training code, I refer to this dataset as final_dataset.csv. 

### Model Creation
Although originally I wanted to use a simple Pytorch neural network with a BERT tokenizer, it was unable to detect meaningful differences between prompts likely to cause hallucinations and those that aren't. So, I switched over to fine-tuning the final layers of a BERT model, which prompted me to compile the larger dataset I described above. I used both Kaggle and Google Colab while developing and testing different models in order to maximize the GPU available for training. My final code can be found at [this colab link](https://colab.research.google.com/drive/1xF6w0pQpxKV26tFF4q2CIHqkMvGTpwLI?usp=sharing). I ended up using LoRA to optimize my training, and after completing a hyperparemeter search over 18 possible combinations I decided on an r-value of 8, an alpha value of 32, and dropout of 0.2 ([here's](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide) a great resource on what these parameters mean). Later, when model learning didn't sufficiently improve, I added the ["key"] layer to the list of trainable layers to increase the model's capacity to adapt to the data and used Claude to create a custom weighted trainer class to counteract my dataset imbalance. I also added a positive weight calculated from the class imbalance, then added a multiplier to increase positive predictions (will cause hallucination) since I chose to prioritize catching the majority of true positives at the expense of also detecting many false positives (I'll talk about this more in the results section).  

### Resources
I used articles such as [this one](https://colab.research.google.com/drive/1xF6w0pQpxKV26tFF4q2CIHqkMvGTpwLI?usp=sharing) on dataframes and dataloaders, [this one](https://medium.com/biased-algorithms/a-practical-guide-to-implementing-early-stopping-in-pytorch-for-model-training-99a7cbd46e9d) for implementing early stopping, and [this video](https://www.youtube.com/watch?v=3M2Gmuh5mtI) for understanding precision, recall, and f1 score. One of the most valuable resources was definitely [this video](https://www.youtube.com/watch?v=4QHg8Ix8WWQ) on fine-tuning BERT for text classification, since the video's task aligns closely with my own. Finally, I used Claude to write code for the LoRA hyperparameter grid search, adapt my training_metrics function, create a weighted trainer (mentioned in the model creation section), enahnce my list of training arguments to improve model performance, and implement early stopping. I cited all AI-generated code in my final Colab link [here](https://colab.research.google.com/drive/1xF6w0pQpxKV26tFF4q2CIHqkMvGTpwLI?usp=sharing)! 


## Results
My best results came from training with 3 LoRA layers and a 1.3 multiplier on the positive weight. Here are my results from Kaggle: 

| Epoch | Validation Loss | Accuracy | Precision | Recall | F1 | T Pos | F Pos | F Neg | T Neg |
|-------|-----------------|----------|-----------|--------|----|-------|-------|-------|-------|
| 1 | 0.685910 | 0.553459 | 0.297787 | 0.657778 | 0.409972 | 148 | 349 | 77 | 380 |
| 2 | 0.632393 | 0.545073 | 0.313725 | 0.782222 | 0.447837 | 176 | 385 | 49 | 344 |
| 3 | 0.617831 | 0.615304 | 0.334884 | 0.640000 | 0.439695 | 144 | 286 | 81 | 443 |
| 4 | 0.604451 | 0.517820 | 0.311396 | 0.862222 | 0.457547 | 194 | 429 | 31 | 300 |
| 5 | 0.596196 | 0.590147 | 0.340385 | 0.786667 | 0.475168 | 177 | 343 | 48 | 386 |
| 6 | 0.595957 | 0.582809 | 0.335238 | 0.782222 | 0.469333 | 176 | 349 | 49 | 380 |
| 7 | 0.595688 | 0.578616 | 0.338208 | 0.822222 | 0.479275 | 185 | 362 | 40 | 367 |

I selected the model at epoch 7 as my best model since it had a high recall (> 80%) but a higher F1 score than the model at epoch 4. The best model, however, would differ depending on the cost of the methods used to reduce hallucinations when the likelihood of one, as detected by this model, is high. For example, if the prevention method involves checking the accuracy of the information, thereby delaying the model's response to the user's query, then a model with higher precision would be necessary. However, if it only involves adapting the prompt to encourage the LLM to be more careful with its answer, it makes sense to sacrifice precision for a much higher recall (many false positives but most true positives caught). In the end, I looked at the model with the highest F1 score among those with a recall above 80%, which results in very few false negatives (only 40, or 4.19% of total prompts) with less than 50% of positives being false whereas with only a slightly higher recall value, at epoch 4 the model has over 50% false positives. 

Here's another example of training results from Kaggle: 
<img width="1049" height="295" alt="image" src="https://github.com/user-attachments/assets/7e22941f-5dd2-49b3-8680-beb694444b1b" />

Ultimately, an approach like this would be able to successfully flag prompts which make models hallucination-prone, and could be adapted to work with different models (though using different datasets) or for different hallucination reduction strategies (more computationally expensive v. less computationally expensive) through changing the positive weighting (as I did in this example) to encourage higher recall, higher precision, or higher F1. 

## Future Improvements
I would like to design a model with a higher F1 score and which can acheive a higher accuracy. In the future, it would be good to reduce the trade-off between catching true positives and the high occurences of false positives. 

## Codebase Tour
* Colab_for_Generating_Dataset_of_Prompts_with_Binary_Hallucination_Labels.ipynb: downloaded from Colab, my code for generating the final_dataset.csv file. Link to Colab [here](https://colab.research.google.com/drive/1bDYQKSXsFnlV4rk53V5sWFWzzFaQr-aB?usp=sharing)
* * final_dataset.csv: the final dataset generated from the Colab above
* Google_Colab_BERT_Training.ipynb: downloaded from Colab, my code for training the final BERT model. Link to Colab [here](https://colab.research.google.com/drive/1xF6w0pQpxKV26tFF4q2CIHqkMvGTpwLI?usp=sharing)

## About the Course
As I mentioned at the beginning, this is my final project after completing MIT's "Intro to Deep Learning" course. I used the 2025 edition, and the course webpage can be found [here](https://introtodeeplearning.com/). My solutions to the labs can be found in [this](https://github.com/108charlotte/Labs-for-MIT-Intro-to-Deep-Learning) GitHub repository. 
