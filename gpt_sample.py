import numpy as np
import csv
import random
import replicate
import os
import openai
import time


def main():
    filename_test = "esnli_test.csv"
    filename_train = "esnli_train.csv"

    with open(filename_test, encoding='utf8', newline='') as file:
        csvreader = csv.reader(file)
        data_for_test = [row for row in csvreader]
        data_for_test = np.array(data_for_test[1:])


    #Set up Seed and extract inputs from files

    np.random.seed(3)
    np.random.shuffle(data_for_test)
    test_data_label = data_for_test[:,1]
    test_data_prem = data_for_test[:,2]
    test_data_hyp = data_for_test[:,3]
    test_data_exp = data_for_test[:,4]

    random.seed(1)
    prompt_index = random.sample(range(test_data_exp.shape[0]),400)
    count = 0
    prompts = []
    idx = 0
    for i in prompt_index:
        prompt = """
Classify the entailment relation for the task of Natural Language Inference into one of the three labels:
entailment (if the premise entails the hypothesis), contradiction (if the hypothesis contradicts the premise), or neutral (if neither entailment nor contradiction).
Also provide an explanation for your chosen label. Providing both the label and the explanation are mandatory.
Here are a few examples.

Premise: A group of children sitting in chairs with monitors over them.
Hypothesis: The kids are in chairs.
Reason: The kids are in chairs so they can watch the monitors
Label: entailment
###
Premise: A young woman with elaborately designed hair is playing a game on a Game Boy.
Hypothesis: An old man is watching tv.
Reason: A young woman is not an old man
Label: contradiction
###
Premise: A man in a black shirt appears to be jumping in the air holding two boards with a sign saying "stomp" in the background.
Hypothesis: A man in a black shirt protesting.
Reason: Holding boards with a sign that says stomp does not imply protesting.
Label: neutral
###
Premise: The curly man wearing gloves is raking the grass.
Hypothesis: He is working outside.
Reason: The man is raking, which is work, grass, which is outside.
Label: entailment
###
Premise: A black dog runs down a snowy hill with his legs deep in the snow.
Hypothesis: A black dog runs through the snow chasing the cat.
Reason: The dog doesn't have to be chasing a cat.
Label: neutral
###
Premise: Man in boat wearing athletic shorts, a backwards baseball cap and sunglasses unhooks a fish he has caught while fishing.
Hypothesis: His bald head is being burned by the sun.
Reason: If a man is wearing a backwards baseball cap, then his bald head will not be exposed and burned by the sun.
Label: contradiction
###
Premise: There is a group of 8 young children holding sticks while standing in the grass.
Hypothesis: 8 children are standing in the grass.
Reason: The children are occupied with the task of holding sticks while standing in the grass, so they must be in the grass.
Label: entailment
###
Premise: Three people gathering and binding sticks or brush.
Hypothesis: Some people are working hard to gather and bind sticks.
Reason: Not all work is hard.
Label: neutral
###
"""
        prompt += "Premise: "+str(test_data_prem[i])+"\n"
        prompt += "Hypothesis: "+str(test_data_hyp[i])
        response = openai.ChatCompletion.create(
		  model="gpt-4/gpt-3.5-turbo",
		  messages=[
			{
			  "role": "system",
			  "content": prompt},
		  ],
		  temperature=0.75,
		  max_tokens=1200,
		  top_p=1,
		  frequency_penalty=0,
	  	  presence_penalty=0
		)
        count+=1
        if(count == 1):
            results = response["choices"][0]["message"]["content"]
            print(i)
            output = results
            f = 0
            print(output)
            print("\n==================================\n")
            count = 0
            time.sleep(0.5)
            
main()
