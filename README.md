# SO_sentiment_analysis

This Repo represents a small project of **Sentiment Analysis** using a pretrained Transformor model on **StackOverflow (SO)** dataset. 

Dataset: 

The first task was to find a suitable SO dataset where the task can be done. To do so, I went through several paper where sentiment analysis has been done. Below are some research that have studied sentiment analysis on SO data.


1. [Supervised Sentiment Classification with CNNs for
Diverse SE Datasets](https://arxiv.org/pdf/1812.09653) 
2. [Classifying emotions in Stack Overflow and JIRA using a multi-label
approach](https://www.sciencedirect.com/science/article/pii/S0950705120300939)

Both of these studies have use the [A Gold Standard for Emotion Annotation in Stack Overflow](https://arxiv.org/pdf/1803.02300)  dataset ([link](https://github.com/collab-uniba/EmotionDatasetMSR18/tree/master)) which is considered as a `Gold standard` SO Emotion dataset, developed for the study titled [Sentiment Polarity Detection for Software Development](https://arxiv.org/pdf/1709.02984); a study of [Senti4SD](https://github.com/collab-uniba/Senti4SD) - a popular sentiment analysis tool. 

The Dataset is made of 4,800 `questions`, `answers`, and `comments` from Stack Overflow, manually annotated for emotions. It includes six basic emotions, namely `love`, `joy`, `anger`, `sadness`, `fear`, and `surprise`.

Model: 

I used the pretraing  **DistilBERT** model from Huggingface. 
Specially followed the [Tutorial to Finetune a pretrained Model](https://huggingface.co/blog/sentiment-analysis-python) by Huggingface. 

Steps to run this project:

1. Clone the repo : ```git clone https://github.com/saimaafrin/SO_sentiment_analysis.git```

2. Run the script named `new_SO_analysis.py` to finetune and save the model. This script includes the code for data processing, model training and testing and saving the model.

```python3 new_SO_analysis.py```

3. Run the script named `SO_analysis_inf.py` to run the inference on new data. 

```python3 SO_analysis_inf.py ```
