import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys
sys.path.append("..")


import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

from prover.lean.verifier import Lean4ServerScheduler
import pickle


def main():
    lean4_scheduler = Lean4ServerScheduler(max_concurrent_requests=130, timeout=300, memory_limit=-1, name='verifier')

    with open("best_of_n_samples_imo.pkl", "rb") as f:
        all_samples = pickle.load(f)

    print(len(all_samples))
    print(len(all_samples[0]))

    verified_outputs = [[] for _ in range(len(all_samples))]

    num_attempts = len(all_samples[0])
    for attempt_idx in tqdm(range(num_attempts), desc="Verifying Attempt", position=0):
        attempts = [x[attempt_idx] for x in all_samples]

        cleaned_attempts = []
        # Map from index in lean verifier output to index in attempts
        cleaned_attempts_ids = []
        for idx, attempt in enumerate(attempts):
            match = re.search(r'```lean4\n(.*?)\n```', attempt, re.DOTALL)
            if match is not None:
                # Output was not well formatted
                cleaned_attempts.append(match.group(1))
                cleaned_attempts_ids.append(idx)

        request_id_list = lean4_scheduler.submit_all_request(cleaned_attempts)
        # outputs_list is len(all_samples long)
        outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)

        for j in range(len(all_samples)):
            if j not in cleaned_attempts_ids: 
                verified_outputs[j].append(None)
            else:
                # Map from index in lean verifier output to index in attempts
                index = cleaned_attempts_ids.index(j)
                verified_outputs[index].append(outputs_list[index])

        # Pickle results
        with open("verified_outputs_imo.pkl", "wb") as f:
            pickle.dump(verified_outputs, f)

    lean4_scheduler.close()

if __name__ == "__main__":
    main()