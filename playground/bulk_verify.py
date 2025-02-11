import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys
sys.path.append("..")


import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
import time
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

from prover.lean.verifier import Lean4ServerScheduler
import pickle


NUM_EXAMPLES = 100

def main():
    
    concurrent_requests = 100
    batch_size = 1

    with open("results/her_imo_4_outputs.pkl", "rb") as f:
        all_samples: List[Dict] = pickle.load(f)

    lean4_scheduler = Lean4ServerScheduler(
        max_concurrent_requests=concurrent_requests,
        batch_size=batch_size,
        timeout=100,
        memory_limit=-1,
        name='verifier'
    )

    goal_theorems = [x["generation"] for x in all_samples if x["type"] == "goal"]
    premise_theorems = [x["generation"] for x in all_samples if x["type"] == "premise"]
    goal_theorems_cleaned = [re.search(r'```lean4\n(.*?)\n```', x, re.DOTALL).group(1) for x in goal_theorems]
    premise_theorems_cleaned = [re.search(r'```lean4\n(.*?)\n```', x, re.DOTALL).group(1) for x in premise_theorems]

    request_id_list = lean4_scheduler.submit_all_request(goal_theorems_cleaned)
    goal_theorem_outputs = lean4_scheduler.get_all_request_outputs(request_id_list)

    request_id_list = lean4_scheduler.submit_all_request(premise_theorems_cleaned)
    premise_theorem_outputs = lean4_scheduler.get_all_request_outputs(request_id_list)

    lean4_scheduler.close()

    result_dict = {
        "goal_theorems": goal_theorem_outputs,
        "premise_theorems": premise_theorem_outputs
    }

    with open("results/her_imo_4_outputs_verified.pkl", "wb") as f:
        pickle.dump(result_dict, f)


if __name__ == "__main__":
    main()