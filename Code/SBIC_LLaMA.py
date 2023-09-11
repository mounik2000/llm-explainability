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

    #Set up Seed and extract inputs from files

    filename_test = "sbic_test.csv"
    filename_train = "sbic_train.csv"

    with open(filename_test, encoding='utf8', newline='') as file:
        csvreader = csv.reader(file)
        data_for_test = [row for row in csvreader]
        data_for_test = np.array(data_for_test[1:])

    np.random.seed(1)
    np.random.shuffle(data_for_test)
    test_data_label = data_for_test[:,2]
    test_data_post = data_for_test[:,1]
    test_data_exp = data_for_test[:,-1]

    random.seed(1)
    prompt_index = range(test_data_label.shape[0])
    count = 0
    prompts = []
    idx = 0
    for i in prompt_index:
        prompt = """
Classify the following posts into offensive or not offensive and generate an explanation for the chosen answer.
Both the answer and explanation are mandatory.
Here are a few examples.

Post: holy fucking hell this sub sucks now . does anyone come up with good original jokes anymore or are you retards still entertained by seeing the same old predictable jokes ?
Explanation: this post implies that reddit is stupid
Answer: offensive
###
Post: how do chinese parents name their children ? throw silverware down the stairs .
Explanation: this post implies that asians can't name their kids properly
Answer: offensive
###
Post: I still remember the last words of my friend who died in the 9/11 incident. "Allah - hu - Akbar"...!
Explanation: this post implies that muslim folks are barbaric
Answer: offensive
###
Post: how do you kill a redneck ? wait until he fucks his sister then cut the breaks to his house
Explanation: this post implies that incest
Answer: offensive
###
Post: Don't play both sides of the fence. Either you fucking with me or you not.
Explanation: this post does not imply anything offensive
Answer: not offensive
###
Post: why should n't you buy a woman a watch ? there 's a clock on the stove
Explanation: this post implies that women belong in the kitchen
Answer: offensive
###
Post: What is the best thing about altzheimers? Making a new friend everyday.
Explanation: this post implies that altzheimers patients forget everything.
Answer: offensive
###
Post: There's a lot of big booty bitches at pharr gym &#128563;&#128584;&#127825;
Explanation: this post does not imply anything offensive
Answer: not offensive
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
###|\n"""
        prompt += "Post: "+str(test_data_post[i])
        prompts.append(prompt)
        count+=1
        if(count == 1):
            results = generator.generate(
                prompts, max_gen_len=1500, temperature=temperature, top_p=top_p
            )
            for result in results:
                print(i)
                output = result.split('###|\n')[1]
                phle = output.split('\n')
                print(phle[0])
                if len(phle)>=3:
                    print(phle[1]+"\n"+phle[2])
                print("\n==================================\n")
            prompts = []
            count = 0        

if __name__ == "__main__":
    fire.Fire(main)

