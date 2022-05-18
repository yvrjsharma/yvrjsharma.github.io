<p align="center">
  <img src="https://user-images.githubusercontent.com/48665385/169143218-26bb7585-e2e9-4a71-b7a2-5f281670201d.png" />
</p>
I came across this wonderful and intriguing Twitter thread from Peter Welinder (VP Product, OpenAI) about GPT3. This fascinating thread from Peter (Twitter handle @npew )
pushed me to use #EleutherAI GPTJ-6B (hosted on @HuggingFace) for a smaller and similar experiment of my own and the results are highly comparable. 
In my humble opinion, this ‘smaller’ 6 Billion parameter model might be comparable in performance to mighty 175 Bilion parameter GPT-3.

Here I have tried using GPTJ to find words within a random mix of letters. I suppose this is not as easy a task for LLM as it is for humans. Some samples are -

![image](https://user-images.githubusercontent.com/48665385/169142118-b12078ac-44ec-482c-9629-d832f8522ca5.png)

Mere prompting the GPTJ-6B model was not enough in the given case. As you can see below the model was unable to spot ‘insomnia’ and ‘protest’ even after increasing number of examples in prompt -

![image](https://user-images.githubusercontent.com/48665385/169142408-a8d3804a-6958-42f5-9c55-7c62cf82971d.png) ![image](https://user-images.githubusercontent.com/48665385/169142437-2956ad87-736d-43b6-bd35-cc95611d2f71.png)

However, following the “Teaching GPT to Reason” approach advocated by Peter in his GPT3 post and keeping in mind the effects of tokenization, I was able to prompt GPTJ-6B to catch the correct words successfully -

![image](https://user-images.githubusercontent.com/48665385/169142765-54741a75-cd1c-40f8-9492-43bfe73c5ff7.png) ![image](https://user-images.githubusercontent.com/48665385/169142807-c4ff7d7c-244d-49e3-a9e5-71e5baa27bb4.png)

Though this was a very small experiment with some cherrypicked examples, my takeaway is that large language models can actually be 'taught' in their own ways. 

By intuitively understanding their operating logic we can break down a problem in steps and guide the model towards the correct answer. 
