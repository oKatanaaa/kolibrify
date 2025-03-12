import json
import argparse
import os
import sys
import time
from typing import List, Dict, Optional, Any

from kolibrify.core.data_consts import ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT
from kolibrify.sft.config import load_training_config
from kolibrify.inference.vllm_model import VllmModel


def load_chat_model(config, temperature: float = 0.7, top_p: float = 0.95, max_tokens: int = 4096, gpu_memory_utilization: float = 0.9):
    """
    Load a VllmModel for chat.
    
    Args:
        config: Config object with model paths
        temperature: Temperature for sampling
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        gpu_memory_utilization: Percentage of GPU memory to utilize
        
    Returns:
        VllmModel instance
    """
    model_path = os.path.join(config.output_dir, "merged")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path '{model_path}' does not exist. Make sure you've merged the model first.")
    
    print(f"Loading model from {model_path}...")
    model = VllmModel(
        merged_model_path=model_path,
        temp=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_model_len=config.max_ctx_len,
        gpu_memory_utilization=gpu_memory_utilization,
        use_tqdm=False,
        enforce_eager=True
    )
    print("Model loaded successfully!")
    return model


def load_default_context(file_path: str) -> List[Dict[str, str]]:
    """
    Load a default conversation context from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing the default context
        
    Returns:
        List of message dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or 'messages' not in data:
            print(f"Error: Default context file must contain a 'messages' key")
            return []
        
        messages = data['messages']
        if not isinstance(messages, list):
            print(f"Error: 'messages' must be a list")
            return []
        
        # Validate format
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                print(f"Error: Each message must have 'role' and 'content' keys")
                return []
        
        return messages
    except Exception as e:
        print(f"Error loading default context: {e}")
        return []


def get_multiline_input(prompt: str = "") -> str:
    """
    Get multi-line input from the user.
    
    Input is terminated by typing '##' on a line by itself.
    Special commands like 'back()', 'reset()', 'exit()' are recognized 
    even if not on a line by themselves.
    
    Args:
        prompt: Prompt to display to the user
        
    Returns:
        User input as a string
    """
    print(prompt)
    print("(Type '##' on a line by itself to submit, or use commands like 'back()', 'reset()', 'exit()')")
    
    lines = []
    while True:
        try:
            line = input("> " if not lines else "... ").rstrip()
            
            # Check for special commands first (allowing immediate execution)
            if line.lower() in ('back()', 'reset()', 'exit()'):
                return line.lower()
            
            # Check for termination sequence
            if line == '##':
                break
            
            lines.append(line)
        except EOFError:
            # Handle Ctrl+D gracefully
            print("\nInput terminated")
            break
        except KeyboardInterrupt:
            # Handle Ctrl+C by canceling the current input
            print("\nInput canceled")
            return ""
    
    # Join the lines with newlines to create the full message
    full_input = "\n".join(lines)
    return full_input


def stream_tokens(conversation: Dict[str, Any], model: VllmModel) -> str:
    """
    Emulate streaming by generating the full response and then printing it token by token.
    
    Args:
        conversation: Conversation dictionary with messages
        model: Model instance to use for generation
        
    Returns:
        Generated response text
    """
    # Get the full response first
    responses = model.predict([conversation])
    response = responses[0]
    
    # Extract the content
    content = response['content']
    
    # Simulate streaming by printing character by character
    for char in content:
        print(char, end='', flush=True)
        # Small delay to simulate token generation
        time.sleep(0.01)
    
    print()  # Add a newline
    return content


def interactive_chat(
    model: VllmModel,
    temperature: float = 0.7,
    top_p: float = 0.95,
    default_context_path: Optional[str] = None
):
    """
    Run an interactive chat session with the model in the terminal.
    
    Args:
        model: The vLLM model instance
        temperature: Temperature for generation
        top_p: Top-p sampling parameter
        default_context_path: Path to a JSON file containing a default conversation context
    """
    # Initialize conversation with default context if provided
    default_context = []
    if default_context_path:
        default_context = load_default_context(default_context_path)
        if default_context:
            print(f"Loaded default context with {len(default_context)} messages")
    
    def start_new_chat():
        """Start a new chat session and get the system message"""
        if default_context:
            # Use the default context
            return default_context.copy()
        else:
            # Get system message using multi-line input
            system_content = get_multiline_input("\nEnter a system message (or press Enter to skip):")
            
            messages = []
            if system_content and system_content not in ('back()', 'reset()', 'exit()', 'regen()'):
                messages.append({"role": ROLE_SYSTEM, "content": system_content})
            
            return messages

    # Start the initial chat
    messages = start_new_chat()
    
    # Main chat loop
    while True:
        # Get user input (multi-line)
        user_input = get_multiline_input("\nEnter your message:")
        
        # Handle empty input
        if not user_input:
            print("Empty input. Please try again.")
            continue
        
        # Handle special commands
        if user_input == 'exit()':
            break
        elif user_input == 'reset()':
            print("\nResetting chat...")
            messages = start_new_chat()
            continue
        elif user_input == 'back()':
            if len(messages) >= 2:  # Remove last assistant and user message
                messages = messages[:-2] if len(messages) >= 2 else []
                print("\nWent back to previous state. Last exchange removed.")
            else:
                print("\nNothing to go back to.")
            continue
        elif user_input == 'regen()':
            if len(messages) > 0 and messages[-1]['role'] == ROLE_ASSISTANT:
                # Remove the last assistant message to regenerate it
                messages.pop()
                if len(messages) > 0 and messages[-1]['role'] == ROLE_USER:
                    print("\nRegenerating the response...")
                    # We'll continue the flow to regenerate the response for the last user message
                else:
                    print("\nNo user message to regenerate response for.")
                    continue
            else:
                print("\nNo response to regenerate.")
                continue
        
        # Add user message
        messages.append({"role": ROLE_USER, "content": user_input})
        
        # Create a conversation object for the model
        conversation = {"messages": messages.copy()}
        
        # Generate and stream the response
        try:
            print("\nAssistant: ", end="", flush=True)
            response_content = stream_tokens(conversation, model)
            
            # Add assistant response to the conversation history
            messages.append({"role": ROLE_ASSISTANT, "content": response_content})
        except Exception as e:
            print(f"\nError generating response: {e}")
            # Remove the user message since we couldn't generate a response
            messages.pop()


def main(config_path, checkpoint, temperature, top_p, max_output_tokens, gpu_id, default_context, gpu_memory_utilization):
    """Main entry point for the chat interface"""
    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Load configuration
    _, config = load_training_config(config_path)
    if checkpoint:
        config.output_dir = os.path.join(config.output_dir, checkpoint)
    
    # Load model directly using VllmModel
    model = load_chat_model(
        config=config,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_output_tokens,
        gpu_memory_utilization=gpu_memory_utilization
    )
    
    # Start interactive chat
    try:
        interactive_chat(
            model=model,
            temperature=temperature,
            top_p=top_p,
            default_context_path=default_context
        )
    except KeyboardInterrupt:
        print("\nChat session ended by user.")
    finally:
        print("\nGoodbye!")


def run():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Start an interactive chat session with a fine-tuned model")
    parser.add_argument("config_path", help="Path to the configuration YAML file", default=None)
    parser.add_argument("--checkpoint", help="Load a certain checkpoint")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling (0.0-1.0)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling parameter (0.0-1.0)")
    parser.add_argument("--max_output_tokens", type=int, default=4096, help="Maximum number of tokens to generate")
    parser.add_argument("--gpu_memory_util", type=float, default=0.9, help="Percentage of GPU memory to utilize")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--default_context", type=str, default=None, 
                        help="Path to a JSON file containing a default conversation context")
    
    args = parser.parse_args()
    
    main(
        config_path=args.config_path,
        checkpoint=args.checkpoint,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        gpu_id=args.gpu_id,
        default_context=args.default_context,
        gpu_memory_utilization=args.gpu_memory_util
    )


if __name__ == "__main__":
    run()