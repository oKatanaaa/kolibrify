import vllm
import multiprocessing
from multiprocessing import Queue
import os

from kolibrify.core.data_utils import ChatMLFormatter


def format_prompts_vllm(messages):
    # tokenizer and max_len parameters are not used here, so can use None
    formatter = ChatMLFormatter(tokenizer=None, max_len=None)
    prompts = []
    for conv in messages:
        prompt = formatter.format_chatml(conv)
        # Add prefix for assistant response
        prompt += '\n<|im_start|>assistant\n'
        prompts.append(prompt)
    return prompts


def split_list(lst, n):
    """Split a list into n sublists of approximately equal size"""
    sublists = []
    len_lst = len(lst)
    chunk_size = len_lst // n  # Size of each sublist
    for i in range(n):
        start = i * chunk_size
        end = start + chunk_size
        sublists.append(lst[start:end])
    # Add remaining elements to the last sublist
    sublists[-1].extend(lst[end:])
    return sublists


class VllmModel:
    def __init__(self, merged_model_path: str, temp=0.0, top_p=0.95, min_p=0, max_tokens=4096, max_model_len=4096, use_tqdm=True, gpu_memory_utilization=0.9, enforce_eager=False):
        self.vllm_model = vllm.LLM(merged_model_path, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization, enforce_eager=enforce_eager)
        self.sampling_params = vllm.SamplingParams(temperature=temp, top_p=top_p, max_tokens=max_tokens, min_p=min_p)
        self.use_tqdm = use_tqdm

    def predict(self, convs):
        prompts = format_prompts_vllm(convs)
        responses = self.vllm_model.generate(prompts=prompts, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm)
        # Extract responses
        openai_responses = []
        for r in responses:
            openai_response = {
                'role': 'assistant',
                'content': r.outputs[0].text
            }
            openai_responses.append(openai_response)
        return openai_responses


class VllmModelWorker(multiprocessing.Process):
    def __init__(self, data_queue: Queue, result_queue: Queue, merged_model_path: str, gpu_id: int, temp=0.0, top_p=0.95, min_p=0, max_tokens=4096, max_model_len=4096):
        """
        Runs on a single GPU in a separate process.
        Contains a single instance of VllmModel and uses it to generate predictions.
        """
        super().__init__()
        self.data_queue = data_queue
        self.result_queue = result_queue
        self.merged_model_path = merged_model_path
        self.gpu_id = gpu_id

        self.temp = temp
        self.top_p = top_p
        self.min_p = min_p
        self.max_tokens = max_tokens
        self.max_model_len = max_model_len

    def run(self):
        os.environ.update({'CUDA_VISIBLE_DEVICES': str(self.gpu_id)})
        self.model = VllmModel(
            self.merged_model_path, 
            temp=self.temp,
            top_p=self.top_p,
            min_p=self.min_p,
            max_tokens=self.max_tokens,
            max_model_len=self.max_model_len
        )
        # Put True into result queue to signal the worker is initialized
        self.result_queue.put(True)
        while True:
            convs = self.data_queue.get()
            if convs is None:
                break
            responses = self.model.predict(convs)
            self.result_queue.put(responses)


class VllmModelDistributed:
    def __init__(self, merged_model_path: str, gpus: list, temp=0.0, top_p=0.95, min_p=0, max_tokens=4096, max_model_len=4096):
        """
        Creates a vLLM worker per GPU. Uses queues to send chats/receive responses.
        """
        # Otherwise the following error is raised:
        # Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
        multiprocessing.set_start_method("spawn")
        self.workers = []
        for gpu_id in gpus:
            self.workers.append(VllmModelWorker(
                Queue(),
                Queue(),
                merged_model_path,
                gpu_id,
                temp=temp,
                top_p=top_p,
                min_p=min_p,
                max_tokens=max_tokens,
                max_model_len=max_model_len
            ))

        self.working = False
    
    def init(self):
        # Start initializing the workers
        for w in self.workers:
            w.start()
        
        # Check all the workers are initialized
        for w in self.workers:
            assert w.result_queue.get() == True, f'Failed to initialize worker at gpu={w.gpu_id}'
        
        self.working = True
    
    def predict(self, convs):
        self._check_working()
        if len(convs) < len(self.workers):
            # Not enough data to load on all workers, use a single worker
            return self.predict_little(convs)
        
        conv_splits = split_list(convs, len(self.workers))
        
        for conv_split, w in zip(conv_splits, self.workers):
            w.data_queue.put(conv_split)
        
        results = []
        for w in self.workers:
            results.extend(w.result_queue.get())
        
        return results

    def predict_little(self, convs):
        """
        Used when there are less data samples than workers.
        Uses a single worker to do all of the predictions.
        """
        self._check_working()

        w = self.workers[0]
        w.data_queue.put(convs)
        return w.result_queue.get()

    def finalize(self):
        for w in self.workers:
            w.data_queue.put(None)

        for w in self.workers:
            w.join()

        self.working = False

    def _check_working(self):
        assert self.working, 'The workers have been terminated or not launched yet.'