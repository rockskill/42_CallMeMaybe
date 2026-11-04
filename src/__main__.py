from llm_sdk import Small_LLM_Model
from src.encoder import Encoder
from src.llm import LLM
from src.callmemaybe import CallMeMaybe
import json


def create_encoder(vocab_path: str) -> Encoder:
    # Parses tokens from the file by the path provided.
    # Then creates the encoder instance with loaded tokens.
    with open(vocab_path, 'r', encoding='utf-8') as f:
        tokens = json.load(f)
    return Encoder(tokens)


if __name__ == "__main__":
    try:
        # Create the CallMeMaybe object as the function processor
        # for functions from the prompt.
        # The function processor contains the LLM and function
        # definitions read from the input JSON file.
        # The LLM consists of provided Qwen 3.8 language model
        # and encoder that encodes the words from the vocabulary.
        llm_model = Small_LLM_Model()
        encoder = create_encoder(llm_model.get_path_to_vocab_file())
        llm = LLM(llm_model, encoder)
        callmemmaybe = CallMeMaybe(llm, args_functions_definition)

        # Load prompts and prepare the output file for the
        # decoded function calls.
        prompts = None
        with open(input_file) as requests:
            prompts = [t['prompt'] for t in json.load(requests)]
        output = open(output_file, 'w')
        output.write('[\n')
        # Process loaded input prompts one by one using CallMeMaybe object.
        for i, prompt in enumerate(prompts):
            if i < len(prompts) - 1:
                output.write(callmemmaybe.process_function(prompt) + ',\n')
            else:
                output.write(callmemmaybe.process_function(prompt) + '\n')
        output.write(']')
        output.close()

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e.msg} " +
              f"at line {e.lineno} column {e.colno}")

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
