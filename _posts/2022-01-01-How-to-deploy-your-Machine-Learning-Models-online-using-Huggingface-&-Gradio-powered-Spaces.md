I built a little ML powered app over the Christmas holidays, hope you are going to enjoy it as much as I did.

## Huggingface
HuggingFace is a startup in the AI field, and there mission is to democratize good machine learning. Its an AI community trying to build the future in which everyone has equal opportunity and access to benfits of latest advances in AI. You can either browse their [model hub](https://huggingface.co/models) to discover, experiment and contribute to new sota models, for example, [gooogle-tapas](https://huggingface.co/google/tapas-base), [distilbert](https://huggingface.co/distilbert-base-uncased), [facebook-wav2vec2](https://huggingface.co/facebook/wav2vec2-base-960h), and so on. Or you can directly use their inference API to serve your moddels directly from HF infrastructure. The most famous artifact that HF has created so far is their Transformer library, which started as an nlp library but now has support for other modalities as well. Now, Transformer provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio.

## Gradio 
[Gradio](https://gradio.app/) is the fastest and easiest way to demo your ML models with a very friendly and feature-rich UI. Almost anyone can us it without a manual and with just a little intuition. You can install Gradio library easily using pip install. I used both Hugging Face and Gradio on Colab so installations were allthemore starightforward and easier. You can deploy your ML model online using just 5-10 code lines depending on the complexity of your implementation. Recently, Gradio has made it possible to embed your deployed ML model to any webpage of your choice. I have done the same at the end of this article, check it out. Gadio code helps you generate a public link for your deployed ML model/app which you can then share with your friends, colleagues at work or a potential employer or collaborator.  

## What I Built
I built a fun project in last couple days using HuggingFace and Gradio functionalities. This project employs mage analysis, language translation and OCR techniques. A user can select an image of his choice with some english text over it as an input. For example, an image with some motivational text written over it like the ones we all receive in our family whatsapp groups all the time. He then gets to make a selection from the given 7 languages as the output language - German, Spanish, French, Turkish, Hindi, Arabic, and Irish. The app then outputs the same image as input but with text now translated in the language selected by the user.

## How I Built It
I am using pytesseract to perform the OCR on input image. Once I have the text 'extracted' from the input image, I employ HuggingFace transformers library to get the desired translation model and tokenizer loaded for an inference. These translation models are open sourced by the [Language Technology Research Group at the University of Helsinki](https://blogs.helsinki.fi/language-technology/), and you can access their account page and pre-trained  odels on HuggingFace'e [website](https://huggingface.co/Helsinki-NLP). The extracted text is then translated into the selected language. For example, if you have selected the language as German, the app will load the "Helsinki-NLP/opus-mt-en-de" translation model from transformers hub and would tranlate the OCR extracted English text to German.

Next, I am using [Kers-OCR](https://github.com/faustomorales/keras-ocr) library to extract the cordinates of English text from the original input image. This library is based on Keras CRNN or [Keras implementation of Convolutional Recurrent Neural Network for Text Recognition](https://github.com/janzd/CRNN). Once I have these cordinates, I perform a cleaning of text using OpenCV Pillow library with just couple lines of code. This cleaning is inspired from [Carlo Borella's incredible post](https://towardsdatascience.com/remove-text-from-images-using-cv2-and-keras-ocr-24e7612ae4f4).

After this, next step is to copy the translated text onto the 'cleansed' image. Current implementation does not take care of pasting the translated text exactly in place of the original English text, however i have plans to do that and more in my next iterations. 

## How You Can Access It
My HuggingFace - Gradio app can be accessed on my account page on thier website, its accessible to public and is available over here - [Translate English Text to Your Regional Language In Your Forwarded Images](https://huggingface.co/spaces/ysharma/TranslateQuotesInImageForwards).
Providing the demo in form of an animation below.
![](/images/20211223_064321.gif)
  
 
## Conclusion 
### Benefits
[HuggingFace Spaces](https://huggingface.co/spaces) is a cool new feature, where anyone can host their AI models using two awesome SDKs - either [Streamlit](https://streamlit.io/) and Gradio. Spaces are a simple way to host ML demo apps directly on your HF profile page or your organization’s profile. This empowers us to create our own little ML project portfolio, showcasing our projects at conferences, stakeholders, or interested parties and work collaboratively with other people in the ML ecosystem.

### Ease of use options, people, future, lastmile ai, deployment, productionised 
>>>>to be added>
