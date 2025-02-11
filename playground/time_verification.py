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

from prover.lean.verifier import Lean4ServerScheduler
import pickle


NUM_EXAMPLES = 100

def main():
    
    model_name = "../../models/deepseek-prover-RL"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    collect_tokens = True
    token_lengths = []
    total_verification_times = {}  # Will now store (concur_requests, batch_size) tuple as key
    launch_server_times = {}       # Will now store (concur_requests, batch_size) tuple as key
    
    # Define parameter ranges to test
    concurrent_requests = [50, 80, 100]
    batch_sizes = [1, 10, 100]

    for concur_requests in concurrent_requests:
        for batch_size in batch_sizes:

            print("--------------------------------")
            print(f"Concurrent requests: {concur_requests}, Batch size: {batch_size}")
            print("--------------------------------")

            with open("results/best_of_n_samples_imo.pkl", "rb") as f:
                all_samples = pickle.load(f)

            samples_cut = []
            for question in all_samples:

                first_attempt = question[0]
                # Collect token lengths
                if collect_tokens:
                    token_lengths.append(len(tokenizer.encode(first_attempt)))

                for attempt in question:
                    samples_cut.append(attempt)
                    if len(samples_cut) >= NUM_EXAMPLES:
                        break
                if len(samples_cut) >= NUM_EXAMPLES:
                    break

            if collect_tokens:
                collect_tokens = False

            # Clean samples
            cleaned_all_samples = []
            for i in range(len(samples_cut)):
                match = re.search(r'```lean4\n(.*?)\n```', samples_cut[i], re.DOTALL)
                if match is not None:
                    cleaned_all_samples.append(match.group(1))

            start_time = time.time()

            launch_server_start_time = time.time()
            lean4_scheduler = Lean4ServerScheduler(
                max_concurrent_requests=concur_requests,
                batch_size=batch_size,
                timeout=100,
                memory_limit=-1,
                name='verifier'
            )
            launch_server_end_time = time.time()

            request_id_list = lean4_scheduler.submit_all_request(cleaned_all_samples)
            outputs_list = lean4_scheduler.get_all_request_outputs(request_id_list)
            lean4_scheduler.close()

            end_time = time.time()

            # ----- Analyze the verification time data ----- #
            print(f"Time taken: {end_time - start_time} seconds")
            print(f"Time taken to launch server: {launch_server_end_time - launch_server_start_time} seconds")
            total_verification_times[(concur_requests, batch_size)] = end_time - start_time
            launch_server_times[(concur_requests, batch_size)] = launch_server_end_time - launch_server_start_time

            # Get success rate
            correct = [x["complete"] for x in outputs_list]
            print(f"Success rate: {sum(correct) / len(correct)}")
            # Get verification times
            verification_times = [x["verify_time"] for x in outputs_list]
            #model_name = "deepseek-ai/DeepSeek-Prover-V1.5-RL"
            # Save histogram plot of verification times

            plt.hist(verification_times, bins=100)
            plt.title(f'Verification Times Distribution\n(Concurrent: {concur_requests}, Batch: {batch_size})')
            plt.savefig(f"./plots/verification_times_c{concur_requests}_b{batch_size}.png")
            plt.close()


    print(f"Token lengths: {token_lengths}")
    print(f"Min token length: {min(token_lengths)}, Mean token length: {sum(token_lengths) / len(token_lengths)}, Max token length: {max(token_lengths)}")

    # Create heatmap of verification times
    plt.figure(figsize=(10, 8))
    times_matrix = [[total_verification_times.get((c, b), 0) 
                    for b in batch_sizes] 
                    for c in concurrent_requests]
    
    plt.imshow(times_matrix, interpolation='nearest')
    plt.colorbar(label='Verification Time (seconds)')
    
    # Add labels
    plt.xticks(range(len(batch_sizes)), batch_sizes)
    plt.yticks(range(len(concurrent_requests)), concurrent_requests)
    plt.xlabel('Batch Size')
    plt.ylabel('Concurrent Requests')
    plt.title('Verification Time Heatmap')
    
    # Add time values as text annotations
    for i in range(len(concurrent_requests)):
        for j in range(len(batch_sizes)):
            plt.text(j, i, f'{times_matrix[i][j]:.1f}', 
                    ha='center', va='center')
    
    plt.savefig("./plots/verification_times_heatmap.png")
    plt.close()

    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(batch_sizes, concurrent_requests)
    Z = np.array(times_matrix)
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    fig.colorbar(surf, label='Verification Time (seconds)')
    
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Concurrent Requests')
    ax.set_zlabel('Verification Time (seconds)')
    ax.set_title('Verification Time Surface Plot')
    
    plt.savefig("./plots/verification_times_surface.png")
    plt.close()

if __name__ == "__main__":
    main()