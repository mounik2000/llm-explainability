import numpy as np
import csv
import random
import replicate

def main():
    filename_test = "sbic_test.csv"
    filename_train = "sbic_train.csv"

    with open(filename_test, encoding='utf8', newline='') as file:
        csvreader = csv.reader(file)
        data_for_test = [row for row in csvreader]
        data_for_test = np.array(data_for_test[1:])

    np.random.seed(2)
    np.random.shuffle(data_for_test)
    test_data_label = data_for_test[:,2]
    test_data_post = data_for_test[:,1]
    test_data_exp = data_for_test[:,-1]

    random.seed(2)
    prompt_index = range(test_data_exp.shape[0])
    count = 0
    prompts = []
    idx = 0
    for i in prompt_index:
        prompt = """
Classify the following posts into offensive or not offensive and generate an answer for the chosen label.
Here are a few examples.

Post: holy fucking hell this sub sucks now . does anyone come up with good original jokes anymore or are you retards still entertained by seeing the same old predictable jokes ?
Explanation: this post implies that reddit is stupid
Answer: offensive
###
Post: how do chinese parents name their children ? throw silverware down the stairs .
Explanation: this post implies that asians can't name their kids properly
Answer: offensive
###
Post: Being Aneroxic is shameful Especially if you're white, you're culturally appropriating African culture. #endculturalappropriation
Explanation: this post implies that black people starve
Answer: offensive
###
Post: Will there be any discussion over the violence in St. Louis, or does only Charlottesville matter?
Explanation: this post does not imply anything offensive
Answer: not offensive
###
Post: Lenin's girlfriend asked him to treat her like a princess He agreed and had her shot.
Explanation: this post is a personal attack
Answer: offensive
###
Post: Why are YouTubers getting famous for clickbait, poronography, drama, &amp; hate speech, &amp; that's stuff that's banned? Oh, wait. We're stupid.
Explanation: this post does not imply anything offensive
Answer: not offensive
###
"""
        prompt += "Post: "+str(test_data_post[i])+"\n"
        prompts.append(prompt)
        count+=1
        if(count == 1):
            results = replicate.run(
    "model",
    input={"prompt": prompt, "max_length": 1200, "temperature": 0.5})
            print(i)
            output = ""
            f = 0
            for result in results:
                if('###' in str(result)):
                    output+=result+" "
                    break
                else:
                    output+=result+" "
            print(output)
            print("\n==================================\n")
            count = 0
            
main()
