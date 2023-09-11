# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import numpy as np
import csv
import random

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    #torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    #print(torch.cuda.memory_summary(device=None, abbreviated=False))
    model = Transformer(model_args)
    #if driver=="cuda":
    torch.set_default_tensor_type(torch.FloatTensor)
    #if driver=="cpu":
    #torch.set_default_tensor_type(torch.HalfTensor)
    model.load_state_dict(checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 1500,
    max_batch_size: int = 1,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    
    filename_test = "esnli_test.csv"

    with open(filename_test, encoding='utf8', newline='') as file:
        csvreader = csv.reader(file)
        data_for_test = [row for row in csvreader]
        data_for_test = np.array(data_for_test[1:])


    #Set up Seed and extract inputs from files

    np.random.seed(1)
    np.random.shuffle(data_for_test)
    test_data_label = data_for_test[:,1]
    test_data_prem = data_for_test[:,2]
    test_data_hyp = data_for_test[:,3]
    test_data_exp = data_for_test[:,4]
    random.seed(1)
    prompt_index = random.sample(range(test_data_exp.shape[0]),1000)
    count = 0
    prompts = []
    idx = 0
    for i in range(0, test_data_exp.shape[0]):
        prompt = """
Classify the entailment relation for the task of Natural Language Inference into one of the three labels:
entailment (if the premise entails the hypothesis), contradiction (if the hypothesis contradicts the premise), or neutral (if neither entailment nor contradiction).
Also provide an explanation for your chosen label. Providing both the label and the explanation are mandatory.
Here are a few examples.

Premise: Two men are using a projector.
Hypothesis: Two men are human.
Label: entailment
Explanation: The two men are humans since only humans know how to use a projector.

Premise: Young girl wearing gray and pink sweater and leather boots shields her face from the camera.
Hypothesis: There is a girl in the photo.
Label: entailment
Explanation: To be in a photo, a person needs to have a picture taken with a camera.

Premise: Young man sitting down eating salad.
Hypothesis: A man sits and eats.
Label: entailment
Explanation: Young man is a man, and eating salad implies eats.

Premise: A black and white dog wearing a face mask runs through a field.
Hypothesis: A dog runs through a field.
Label: entailment
Explanation: A dog runs through a field is a rephrasing of A black and white dog wearing a face mask runs through a field

Premise: Woman in a tennis match is getting ready to hit the approaching tennis ball.
Hypothesis: The woman is playing outdoors.
Label: neutral
Explanation: Playing outdoors does not imply a woman is in tennis match.

Premise: People at an outdoor event at a park.
Hypothesis: A crowd at an outdoor event.
Label: entailment
Explanation: a crowd is made of people.

Premise: Several young people sit at a table playing poker.
Hypothesis: The people have money on the table.
Label: neutral
Explanation: People are playing poker on a table

Premise: A man and a woman walking down a street, carrying luggage.
Hypothesis: A man and woman carrying luggage.
Label: entailment
Explanation: man and women implies man and women. both carrying luggage

Premise: A woman dressed in jeans and a shirt, auditioning for a play, while a man waits for his turn.
Hypothesis: Nobody is holding auditions.
Label: contradiction
Explanation: either somebody would be holding the auditioning or not

Premise: A dog is about to get a ball that is on orange carpet.
Hypothesis: The dog is chasing a ball that isn't his.
Label: neutral
Explanation: Being about to get a ball does not mean one is chasing it; getting that ball doesn't mean it isn't his.

Premise: A person is sitting on the couch, holding a large bottle of alcohol.
Hypothesis: A person is starting to get drunk.
Label: neutral
Explanation: Sentence one doesn't explain that the person is drinking the alcohol and that he is getting drunk.

Premise: An older man in a jacket making cookies.
Hypothesis: An older man is eating a cookie.
Label: contradiction
Explanation: Making cookies and eating them are different actions.

Premise: Man is playing his guitar in the street.
Hypothesis: The man is on top of a tree.
Label: contradiction
Explanation: If the man is in the street, he is not on top of a tree.

Premise: People shopping in a hallway with markets.
Hypothesis: People at a trade show looking at exhibits.
Label: contradiction
Explanation: A markets shopping area is a different type of zone to a trade show

Premise: a person with cutting machine cleaning the garden.
Hypothesis: A person is in their room jumping on the bed
Label: contradiction
Explanation: cleaning is not the same as jumping

Premise: A young girl wearing a blue jacket is stepping into a puddle.
Hypothesis: A young girl is wearing a blue jacket and rain boots.
Label: neutral
Explanation: Stepping into a puddle does not necessarily mean in rain boots.

Premise: three young women goofing off at a bar.
Hypothesis: Three young women at a bar.
Label: entailment
Explanation: Three young women were goofing off at the bar, which means they were at a bar.

Premise: A person wearing a blue jacket, blue jacket and black earmuffs is holding up a newspaper.
Hypothesis: A person is very cold
Label: neutral
Explanation: Just because a person is wearing a jacket does not mean the person is very cold.###\n"""
        prompt += "Premise: "+str(test_data_prem[i])+"\n"
        prompt += "Hypothesis: "+str(test_data_hyp[i])+"\n"
        prompt += "Label: "
        prompts.append(prompt)
        count+=1
        if(count == 1):
            results = generator.generate(
                prompts, max_gen_len=1500, temperature=temperature, top_p=top_p
            )
            for result in results:
                print(i)
                output = result.split('###\n')[1]
                phle = output.split('\n')
                print(phle[0]+"\n"+phle[1])
                if len(phle)>=4:
                    print(phle[2]+"\n"+phle[3])
                print("\n==================================\n")
            prompts = []
            count = 0        

if __name__ == "__main__":
    fire.Fire(main)

