"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests. 
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
"""

import re
import difflib
import json
import os
from textwrap import dedent
from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval

_CITATION = """
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {"humaneval": create_task(True), "humaneval-unstripped": create_task(False)}


def create_task(strip_prompt):
    class HumanEval(GeneralHumanEval):
        def __init__(self, **kwargs):
            super().__init__(strip_prompt, **kwargs)

    return HumanEval


class GeneralHumanEval(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "openai_humaneval"

    def __init__(self, strip_prompt, k=[1, 10, 100], num_workers=16, timeout=3.0):
        super().__init__(
            stop_words=["\nclass", "\ndef", "\n@", "\nprint", "\nif", "\n```", "<file_sep>"],
            requires_execution=True,
        )
        self.strip_prompt = strip_prompt
        self.k = k
        self.num_workers = num_workers
        self.timeout = timeout
        self.parsing_record = []

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset["test"]

    # def get_prompt(self, doc):
    #     """Builds the prompt for the LM to generate from."""
    #     prompt = doc["prompt"].strip() if self.strip_prompt else doc["prompt"]
    #     # signature, nl_prompt, _ = prompt.split('"""')
    #     if '"""' in prompt:
    #         splited = prompt.split('"""')
    #     else:
    #         splited = prompt.split("'''")
    #     instruction = splited[-2]
    #     if self.SUGARIZED:
    #         response_prefix = f"""```sugarized_python\n{prompt}"""
    #     else:
    #         response_prefix = f"""```python\n{prompt}"""
    #     return self.MAGICODER_PROMPT.format(instruction=instruction, response=response_prefix)


    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        prompt = doc["prompt"].strip() if self.strip_prompt else doc["prompt"]
        if self.SUGARIZED:
            if '"""' in prompt:
                splited = prompt.split('"""')
            else:
                splited = prompt.split("'''")
            code = splited[0] + 'pass'
            code = self.PARSER.encode(code)
            prompt = code[:-4] + '"""' + splited[1] + '"""' + splited[2]
        prompt = prompt + '\n'

        return prompt


    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point


    # def postprocess_generation(self, generation, idx):
    #     """Defines the postprocessing for a LM generation.
    #     :param generation: str
    #         code generation from LM
    #     :param idx: int
    #         index of doc in the dataset to which the generation belongs
    #         (not used for Humaneval-Task)
    #     """
    #     prompt = self.get_prompt(self.dataset["test"][idx])
    #     leading_tokens = '```sugarized_python' if self.SUGARIZED else '```python'
    #     index = prompt.find(leading_tokens) + len(leading_tokens)
    #     header = prompt[index:]
    #     generation = self._stop_at_stop_token(generation[len(prompt):], self.stop_words)
    #     generation = header + generation

    #     return generation

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        prompt = self.get_prompt(self.dataset["test"][idx])
        generation = generation[len(prompt):]
        return prompt + self._stop_at_function_end(generation)
    
    def _stop_at_function_end(self, decoded_string):
        # find a new line that starts without indentation
        regex = r"\n[a-z<]"
        match = re.search(regex, decoded_string)
        if match:
            return decoded_string[:match.start()]
        else:
            return decoded_string


    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        results, exec_results = compute_code_eval(
            references=references,
            predictions=generations,
            k=self.k,
            num_workers=self.num_workers,
            timeout=self.timeout,
        )
        # save the parsing record
     
        return results, exec_results
