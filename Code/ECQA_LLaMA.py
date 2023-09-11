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
import jsonlines

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

    filename_test = "ecqa_test.jsonl"
	filename_train = "ecqa_train.jsonl"
	test_data_question = []
	test_data_answer = []
	test_data_choices = []
	test_data_exp = []
	with jsonlines.open(filename_test, 'r') as file:
		for line in file:
		    test_data_question.append(line['question'])
		    test_data_answer.append(line['answer'])
		    test_data_choices.append(', '.join(line['choices']))
		    test_data_exp.append(line['explanation'])

    random.seed(1)
    prompt_index = range(len(test_data_exp))
    count = 0
    prompts = []
    idx = 0
    for i in prompt_index:
        prompt = """
Select the correct answer from the given five choices to the given question. 
Provide an explanation for your chosen answer. Both the answer and the explanation are mandatory.
Here are a few examples.

Question: What do parents tell their kids to stop doing when they want them to go to sleep?
Choices: get in bed, get dance, get into bed, stop talking, lay down
Answer: stop talking
Explanation: Kids generally keep chit chatting in bed without trying to sleep. Chit chatting means talking continuously. When someone keeps chit chatting they are asked to stop talking.

Question: The artificial intelligence was constantly analysing the data, it was programmed to always be what?
Choices: learning more about, healing, better knowledge, enlightened, change
Answer: learning more about
Explanation: Artificial intelligence is a software that is programmed to be intelligent. Being intelligent means being able to learn more about something.

Question: What color is milk?
Choices: fridge, white, chocolate flavored, good for babies, golden
Answer: white
Explanation: Milk is made up of water and other protein forming tiny particles that refract light. The refraction causes light to scatter which makes the milk appear white.

Question: Where can you find a place to eat and places to buy items of many different kinds?
Choices: city, downtown, own house, bar, shopping center
Answer: shopping center
Explanation: A mall is the only place where one can find places to eat and shop. A mall is also called a shopping center.

Question: Dying due to suffocation means you have what?
Choices: getting cold, would cease to exist, no more air, stop breathing, death and decay
Answer: stop breathing
Explanation: Suffocation means unable to breath properly. Breathing stops when you soffocate for long time. Stopped breathing results in death.

Question: Where is an active solider likely to sleep?
Choices: armored car, white house, underwater, tent, trenches
Answer: tent
Explanation: Tent is a thing that is used for sleeping. An active soldier is likely to sleep in a tent.

Question: Where would you find people standing in a line outside?
Choices: bus depot, end of line, opera, neighbor's house, meeting
Answer: bus depot
Explanation: Bus depots are located outside in an open space. People stand at a bus depot to board a bus. People have to board the bus one person at a time. In order to avoid chaos, people usually stand in a line at a bus depot.

Question: Where is a good place to leave a car in public?
Choices: race track, parking lot, in the market place, city, repair shop
Answer: parking lot
Explanation: Cars can be left in a parking lot. Parking lot is a public place. Parking lot is meant for parking.###\n\n"""
		prompt+= "Question: "+str(test_data_question[i])+"\n"
		prompt+= "Choices: "+str(test_data_choices[i])+"\n"
		prompt+= "Answer: "
        prompts.append(prompt)
        count+=1
        if(count == 1):
            results = generator.generate(
                prompts, max_gen_len=1500, temperature=temperature, top_p=top_p
            )
            for result in results:
                print(i)
                output = result.split('###\n\n')[1]
                phle = output.split('\n')
                print(phle[0]+"\n"+phle[1])
                if len(phle)>=4:
                    print(phle[2]+"\n"+phle[3])
                print("\n==================================\n")
            prompts = []
            count = 0        

if __name__ == "__main__":
    fire.Fire(main)

