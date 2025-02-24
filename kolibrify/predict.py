import os
# Required to run the script
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
from glob import glob

from kolibrify.sft.config import load_training_config
from kolibrify.inference import (
    load_model, load_dataset, save_dataset, predict
)


def main(config_path, dataset_path, dataset_save_path='output.jsonl', 
         backend='vllm', type='last', temp=0, top_p=0.95, 
         max_output_tokens=4096, gpus='0'):
    
    _, config = load_training_config(config_path)
    model = load_model(config, backend, temp, top_p, max_output_tokens, gpus)
    print('Loaded model.')
    
    if os.path.isdir(dataset_path):
        # Folder with multiple datasets is provided
        assert not dataset_save_path.endswith('.jsonl'), 'Dataset save path must be a directory when load path is a directory.'
        os.makedirs(dataset_save_path, exist_ok=True)
        print('Received a directory.')

        # Load data
        dataset_paths = glob(os.path.join(dataset_path, '*.jsonl'))
        assert len(dataset_paths) > 0, 'No .jsonl files found in the dataset directory.'
        print('Found', len(dataset_paths), 'datasets.')

        conv_list = []
        for dataset_path in dataset_paths:
            conv_list.append(load_dataset(dataset_path))
        
        # Generate and save predictions
        for dataset_path, conversations in zip(dataset_paths, conv_list):
            print('Generating responses for', dataset_path)
            conversations = predict(
                model=model,
                conversations=conversations,
                type=type
            )
            save_path = os.path.join(dataset_save_path, os.path.basename(dataset_path))
            save_dataset(conversations, save_path)
            print(f'Saved responses for {dataset_path} to {save_path}')
    else:
        # Path to a single dataset file is provided
        conversations = load_dataset(dataset_path)
        
        conversations = predict(
            model=model,
            conversations=conversations,
            type=type
        )
        
        save_dataset(conversations, dataset_save_path)
        print('Saved responses to', dataset_save_path)
    
    model.finalize()

def run():
    parser = argparse.ArgumentParser(description="Generate predictions using a fine-tuned model")
    parser.add_argument("config_path", help="Path to the configuration YAML file")
    parser.add_argument("dataset_path", help="Path to the dataset file or directory containing datasets")
    parser.add_argument("dataset_save_path", help="Path to save the predictions", nargs='?', default='output.jsonl')
    parser.add_argument("--backend", help="Backend to use for inference", default='vllm')
    parser.add_argument("--type", help="Type of inference to perform", default='last')
    parser.add_argument("--temp", help="Temperature for sampling", type=float, default=0)
    parser.add_argument("--top_p", help="Top-p sampling parameter", type=float, default=0.95)
    parser.add_argument("--max_output_tokens", help="Maximum number of tokens to generate", type=int, default=4096)
    parser.add_argument("--gpus", help="Comma-separated list of GPU indices to use", default='0')
    
    args = parser.parse_args()
    main(args.config_path, args.dataset_path, args.dataset_save_path,
         args.backend, args.type, args.temp, args.top_p, 
         args.max_output_tokens, args.gpus)