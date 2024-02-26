import fire
import transformers
from transformers import AutoTokenizer,GenerationConfig
import torch
from peft import PeftModel
from tqdm import tqdm
import os

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with further context. "
        "Write a response that appropriately completes the request.\n\n"
        "Instruction:\n{instruction}\n\n Input:\n{input}\n\n Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "Instruction:\n{instruction}\n\nResponse:"
    ),
    "prompt_with_chat": (
        "<<SYS>>\nBelow is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n <</SYS>>\n\n"
        "[INST] {instruction}[/INST]\n\n Input:\n{input}\n\n Response:"
    )
}

def generate_prompt(instruction, input=None, is_chat=False):
    if input:
        if is_chat:
            return PROMPT_DICT["prompt_with_chat"].format(instruction=instruction,input=input)
        else:
            return PROMPT_DICT["prompt_input"].format(instruction=instruction,input=input)
    else:
        return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


def main(
    base_model: str = "",
    lora_weights: str = "",
    cache_dir: str = "",
):
        # print("use Linear Scaled RoPE...")  
        # from util.llama_rope_scaled_monkey_patch import replace_llama_rope_with_scaled_rope
        # replace_llama_rope_with_scaled_rope()

        model = transformers.AutoModelForCausalLM.from_pretrained(
                            base_model,
                            torch_dtype=torch.bfloat16,
                            # load_in_8bit=True,
                            use_flash_attention_2=True,
                            cache_dir=cache_dir,
                            device_map="auto",
                    )

        if lora_weights != "":
            model = PeftModel.from_pretrained(
                        model,
                        lora_weights,
                        device_map="auto",
                        cache_dir=cache_dir,
                        torch_dtype=torch.bfloat16,
                    )
            print("Finished loading PEFT model")
        tokenizer =  AutoTokenizer.from_pretrained(base_model,use_fast=False,cache_dir=cache_dir)
        tokenizer.pad_token=tokenizer.eos_token
        model.eval()

        is_chat = "chat" in base_model or "Mistral-7B-Instruct" in base_model

        def generator(
                instruction,
                input=None,
                temperature=0.1,
                top_p=0.75,
                top_k=40,
                num_beams=1,
                max_new_tokens=512,
                **kwargs,
        ):

            ins_f = generate_prompt(instruction,input,is_chat)
            inputs  =  tokenizer(ins_f, return_tensors="pt")
            input_ids = inputs["input_ids"].cuda()
            generation_config = GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    num_beams=num_beams,
                    **kwargs,
                )

            # Without streaming
            with torch.no_grad():
                print("Generating!")
                generation_output = model.generate(
                        input_ids=input_ids,
                        generation_config=generation_config,
                        return_dict_in_generate=True,
                        output_scores=False,
                        max_new_tokens=max_new_tokens,
                    )
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
            response = output.split("Response:")[1].strip()
            return response
        print("finish loading model..")

        
        # prompt = """Nvidia led the tech sector higher yesterday as it continues its bull run after KeyBanc Capital Markets raised their price target from $550 to $620 last week. Micron Technologies closed +2.5% higher after gapping up with Apple and Google, both gaining nearly +1%.
            
        # Based on the news, should I buy Nvidia or sell Nvidia stocks?"""
        
        
        def read_by_paragraphs(filename):
            with open(filename, 'r', encoding='utf-8') as file:
                prompts = file.read().split('\n\n')  # Splitting by two newline characters
                return [prompt.replace('\n', ' ') for prompt in prompts]  # Replacing newlines within paragraphs

        # Read the entire file as a long string, then split the string
            
        # sample_text = """
        # [][heading]heading 1[/heading]

        # text 1 

        # [heading]heading 2[/heading]

        # text 2 
        # """

        def read_by_headings(filename):
            with open(filename, 'r', encoding='utf-8') as file:
                text = file.read() # entire file read as a long string
            
            lines = text.strip().split('\n\n')
            paragraphs = []
            current_heading = ''
            current_paragraph = ''
            
            for line in lines:
                line = line.strip()
                if line.startswith('[heading]') and line.endswith('[/heading]'):
                    # Save the previous paragraph if it exists
                    if current_heading and current_paragraph:
                        paragraphs.append(current_heading + ' ' + current_paragraph)
                    
                    # Reset for new heading and paragraph
                    current_heading = line[9:-10]  # Extracting the heading text
                    current_paragraph = ''
                else:
                    # Accumulating lines into the current paragraph
                    current_paragraph += (' ' if current_paragraph else '') + line

            # Save the last paragraph if it exists
            if current_heading and current_paragraph:
                paragraphs.append(current_heading + ' ' + current_paragraph)

            return paragraphs
        
        def read_by_titles_and_headings(filename):
            with open(filename, 'r', encoding='utf-8') as file:
                text = file.read() # entire file read as a long string
            
            lines = text.strip().split('\n\n')
            paragraphs = []
            current_title = ''
            current_heading = ''
            current_paragraph = ''
            title_found = False  # Flag to check if at least one title has been found

            for line in lines:
                line = line.strip()
                if line.startswith('[title]'):
                    current_title = line[7:-8]  # Extract title text
                    title_found = True
                    # Reset heading and paragraph when a new title is found
                    current_heading = ''
                    current_paragraph = ''
                elif line.startswith('[heading]') and line.endswith('[/heading]'):
                    # Save previous heading and paragraph if they exist
                    if title_found and current_heading and current_paragraph:
                        full_text = f"{current_title}: {current_heading} {current_paragraph}"
                        paragraphs.append(full_text)
                    
                    # Reset for new heading and paragraph
                    current_heading = line[9:-10]  # Extracting the heading text
                    current_paragraph = '' # Reset paragraph
                else:
                    if title_found:
                        # Accumulating lines into the current paragraph only if a title has been found
                        current_paragraph += (' ' if current_paragraph else '') + line

            # Save the last heading and paragraph if they exist
            if title_found and current_heading and current_paragraph:
                full_text = f"{current_title}: {current_heading} {current_paragraph}"
                paragraphs.append(full_text)

            return paragraphs

        def read_entire_file(filename):
                with open(filename, 'r', encoding='utf-8') as file:
                     text = file.read()  # Read the entire file content into a single string
                return text
            

        def write_to_file(inf_results, inference):
            with open(inf_results, 'w', encoding='utf-8') as file:
                file.write(inference) 
            
        # I used cp /Users/sunnielee/Desktop/OneDriveUoE/PROJECT/risk_factors/0000001800/2023-02-17.txt /Users/sunnielee/Library/CloudStorage/OneDrive-UniversityofEdinburgh/InvestLM/reports_abbott.txt 
                # to copy the file from the extracted risk factors as the txt to be read. 0000001800 stands for Abbott.   
        reports_filename = 'reports/reports_demo.txt'  # The file with reports selected from 10K reports of companies.
        
        if "falcon_7b_instruct" in base_model:
            results_filename = 'results_txt/falcon_7b_instruct/results_demo.txt'
        elif "gemma" in base_model:
            results_filename = 'results_txt/gemma7b/results_demo.txt'
        elif "investLM" in base_model:
            results_filename = 'results_txt/investLM/results_demo.txt'
        elif "Llama-2-7b-chat" in base_model:
            results_filename = 'results_txt/llama7b_chat/results_demo.txt'
        else:
            results_filename = 'results_txt/results_demo.txt'
        
        # The file with inferred results.

        # Check if the file results_filename exists
        if not os.path.exists(results_filename):
        # If the file does not exist, create a new empty file
            with open(results_filename, 'w') as file:
                pass  # Creating an empty file


        prompt_q = '''Based on the report summary, ss management executing well on their stated strategy? Are they meeting their own goals and projections?''' # question

        # reports = read_by_paragraphs(reports_filename)

        reports = read_entire_file(reports_filename)
        if isinstance(reports, str):
            reports = [reports]
        results = ""

        for paragraph in tqdm(reports):
            
            prompt = paragraph + "\n\n" + prompt_q

            output = generator(instruction = prompt,
                                 input = None,
                                 temperature = 0.1,
                                 top_p = 0.75,
                                 top_k = 40,
                                 num_beams = 1,
                                max_new_tokens = 512)
            
            results += output + "\n\n"
            print(output)

        write_to_file(results_filename, results)

if __name__ == "__main__":
    fire.Fire(main)

  


   
