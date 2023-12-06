import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import gradio as gr
model_name="distilbert-base-uncased-finetuned-sst-2-english"
try:
    model=AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer=AutoTokenizer.from_pretrained(model_name)
except Exception as e:
    print("error loading the model:",e)
    exit()


#definr sentiments ananlysis function

def sentiment_analysis(text):
    try:
        input_ids=tokenizer.encode(text,truncation=True,padding=True,return_tensors="pt")
        output=model(input_ids)[0]
        sentiment=torch.argmax(output).item()
        return "Positive" if sentiment==1 else"negative"
    except Exception as e:
        print("error during sentiment analysis:",e)
        return "netural"
    

#create chatbot interface

def chatbot(user_input):
    sentiment=sentiment_analysis(user_input)
    if sentiment=="positive":
        response="im glad to here that, how can i assist you?"
    else:
        response="im sorry to here that, Is there anything you want to discuss?"
    return response

iface=gr.Interface(fn=chatbot,inputs="text",outputs="text",title="sentiment-Based Chatbot")
iface.launch()


