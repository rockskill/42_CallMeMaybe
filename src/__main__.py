from llm_sdk import Small_LLM_Model
from src.encoder import Encoder
from src.llm import LLM
from src.callmemaybe import CallMeMaybe
import json


def create_encoder(vocab_path: str) -> Encoder:
    with open(vocab_path, 'r', encoding='utf-8') as f:
        tokens = json.load(f)
    return Encoder(tokens)


if __name__ == "__main__":
    try:
        llm_model = Small_LLM_Model()
        encoder = create_encoder(llm_model.get_path_to_vocab_file())
        llm = LLM(llm_model, encoder)
        cmm = CallMeMaybe(llm, functions_definition)

        prompts = None
        with open(input_file) as requests:
            prompts = [t['prompt'] for t in json.load(requests)]
        output = open(output_file, 'w')
        output.write('[\n')
        for i, p in enumerate(prompts):
            if i < len(prompts) - 1:
                output.write(cmm.process_func(p) + ',\n')
            else:
                output.write(cmm.process_func(p) + '\n')
        output.write(']')
        output.close()

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e.msg} " +
              f"at line {e.lineno} column {e.colno}")

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")