import os
from copy import deepcopy

from kolibrify.sft.config import TrainingConfig
from .vllm_model import VllmModel, VllmModelDistributed


def load_model(
    config: TrainingConfig, 
    backend: str = 'vllm',
    temp: float = 0,
    top_p: float = 0.95,
    max_output_tokens: int = 4096,
    gpus: str = '0'
):
    gpus = list(map(int ,gpus.split(',')))
    assert all(map(lambda x: x >= 0, gpus)), f'Some gpu ids are not valid. Received {gpus}'

    model_path = os.path.join(config.output_dir, 'merged')
    assert os.path.exists(model_path), "The model must be merged but is not."
    if backend == 'vllm':
        model = VllmModelDistributed(
            model_path, gpus=gpus, 
            temp=temp, top_p=top_p, max_tokens=max_output_tokens, max_model_len=config.max_ctx_len
        )
    else:
        raise ValueError('At the moment only vllm backend is supported.')
    model.init()
    return model


def predict(
    model,
    conversations: list,
    type: str = 'last',
):
    conversations = deepcopy(conversations)

    responses = model.predict(conversations)
    print('Finished generating responses.')
    
    for conv, response in zip(conversations, responses):
        conv['messages'].append(response)
        
    return conversations