<p align="center">
  <img src="https://user-images.githubusercontent.com/48665385/169143218-26bb7585-e2e9-4a71-b7a2-5f281670201d.png" />
</p>
I came across this wonderful and intriguing Twitter thread from Peter Welinder (VP Product, OpenAI) about GPT3. 

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">GPT-3 is amazing at complex tasks like creative writing and summarizing. But it&#39;s surprisingly bad at reversing words. 🤔<br><br>The reason is that GPT-3 doesn&#39;t see the world the way we humans do. 👀<br><br>If you teach it to reason, it can get around its limitations to get really good. 💡 <a href="https://t.co/Cnd9iN87oq">pic.twitter.com/Cnd9iN87oq</a></p>&mdash; Peter Welinder (@npew) <a href="https://twitter.com/npew/status/1525900849888866307?ref_src=twsrc%5Etfw">May 15, 2022</a></blockquote> 

This fascinating thread from Peter (Twitter handle @npew ) pushed me to use #EleutherAI GPTJ-6B (hosted on @HuggingFace) for a smaller and similar experiment of my own and the results are highly comparable. And in my humble opinion, this ‘smaller’ 6 Billion parameter model might be comparable in performance to mighty 175 Bilion parameter GPT-3.

Here I have tried using GPTJ to find words within a random mix of letters. I suppose this is not as easy a task for LLM as it is for humans. Some samples are -

![image](https://user-images.githubusercontent.com/48665385/169142118-b12078ac-44ec-482c-9629-d832f8522ca5.png)

Mere prompting the GPTJ-6B model was not enough in the given case. As you can see below the model was unable to spot ‘insomnia’ and ‘protest’ even after increasing number of examples in prompt -

![image](https://user-images.githubusercontent.com/48665385/169142408-a8d3804a-6958-42f5-9c55-7c62cf82971d.png) ![image](https://user-images.githubusercontent.com/48665385/169142437-2956ad87-736d-43b6-bd35-cc95611d2f71.png)

However, following the “Teaching GPT to Reason” approach advocated by Peter in his GPT3 post and keeping in mind the effects of tokenization, I was able to prompt GPTJ-6B to catch the correct words successfully -

![image](https://user-images.githubusercontent.com/48665385/169142765-54741a75-cd1c-40f8-9492-43bfe73c5ff7.png) ![image](https://user-images.githubusercontent.com/48665385/169142807-c4ff7d7c-244d-49e3-a9e5-71e5baa27bb4.png)

Though this was a very small experiment with some cherrypicked examples, my takeaway is that large language models can actually be 'taught' in their own ways. 

By intuitively understanding their operating logic we can break down a problem in steps and guide the model towards the correct answer. 

I have also tweeted this in form of a Twitter thread and it can be found over here - 

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">This fascinating thread from <a href="https://twitter.com/npew?ref_src=twsrc%5Etfw">@npew</a> pushed me to use <a href="https://twitter.com/hashtag/EleutherAI?src=hash&amp;ref_src=twsrc%5Etfw">#EleutherAI</a> GPTJ-6B (hosted on <a href="https://twitter.com/huggingface?ref_src=twsrc%5Etfw">@HuggingFace</a> ) for a smaller and similar experiment of my own and the results are comparable. IMO this ‘smaller’ 6B parameter model might also be similarly capable. A Thread -🧵 <a href="https://t.co/eAhDK7ubFq">https://t.co/eAhDK7ubFq</a></p>&mdash; Yuvi (@yvrjsharma) <a href="https://twitter.com/yvrjsharma/status/1526984609568808962?ref_src=twsrc%5Etfw">May 18, 2022</a></blockquote> 


