from flask import Blueprint, Flask, render_template,request
import google.generativeai as genai
import requests
import numpy as np
from datetime import datetime


API_URL = "https://api-inference.huggingface.co/models/renderartist/simplevectorflux"
headers = {"Authorization": "Bearer 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"}
genai.configure(api_key= "AIzaSyA0T-E9wSZngrAqsK1nz2JOVzonSxTxTOQ")
img_ids = []


def stable_diffusion(prompt):
    
   def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.content
   image_bytes = query({
    "inputs": prompt,
    })
    # You can access the image with PIL.Image for example
   import io
   from PIL import Image
   image = Image.open(io.BytesIO(image_bytes))
   image.save('./static/images/saved/img.png')
   print("img saved")






views = Blueprint('views', __name__)


@views.route('/')
def kick_start():
    return render_template('kick_start.html')


@views.route('/home')
def home():
    return render_template('home.html')


@views.route('/input')
def input():
    return render_template('input.html')

@views.route('/output', methods=['POST', "GET"])
def output():
    content = []
    no = 4
    file=open('./static/story.txt', mode='w')
    pro = request.form['userPrompt']
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"{pro} "
    response =model.generate_content(prompt)
    file.write(response.text)
    file.close()
    with open('./static/story.txt', 'r', encoding='utf-8') as file:
       paragraphs = file.read()
    paragraphs = paragraphs.split('\n\n')
    print(paragraphs[0])
    stable_diffusion(paragraphs[0])
    return render_template('output.html',content= paragraphs)