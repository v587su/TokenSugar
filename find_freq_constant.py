import json
from collections import Counter, defaultdict
import ast
import re
import logging
import os
from modifier.pattern import create_patterns, Pattern, apply_patterns 
from sugar_dataset import BaseDataset

from datasets import load_dataset
logging.basicConfig(level=logging.DEBUG)

hf_home = os.environ.get('HF_HOME')
dataset = load_dataset("LimYeri/LeetCode_Python_Solutions_v2")['train']

def extract_content(examples):
    new_examples = []
    for content in examples['content']:
        try:
            # Extract code between the markdown markers
            code = content.split('```python')[1].split('```')[0]
            # Validate code by parsing
            ast.parse(code)
            new_examples.append(code)
        except Exception as e:
            logging.debug(f"Error parsing content: {content}")
            continue
    return {'content': new_examples}

dataset = dataset.map(
    extract_content,
    batched=True,
    desc="Extracting content from LeetCode",
    remove_columns=[c for c in dataset.column_names if c != 'content'],
    num_proc=8
)

def count_constants_in_code(code: str) -> dict:
    """
    Parses Python code and counts the frequency of constant values.

    Args:
        code (str): A string of Python source code.

    Returns:
        Counter: A collections.Counter mapping each constant value to its frequency.
    """
    class ConstantVisitor(ast.NodeVisitor):
        def __init__(self):
            self.counter = defaultdict(int)

        def visit_Constant(self, node):
            # Increment the counter for the encountered constant value
            self.counter[str(node.value)] += 1
            # Continue traversing any children nodes
            self.generic_visit(node)
        
        def visit_List(self, node):
            if len(node.elts) == 0:
                self.counter['[]'] += 1
        
        def visit_Tuple(self, node):
            if len(node.elts) == 0:
                self.counter['()'] += 1
        
        def visit_Dict(self, node):
            if len(node.keys) == 0:
                self.counter['{}'] += 1

    tree = ast.parse(code)
    visitor = ConstantVisitor()
    visitor.visit(tree)
    return dict(visitor.counter)

def add_constant_counts(examples):
    constant_counts = []
    for code in examples['content']:
        try:
            counts = count_constants_in_code(code)
            # Convert the Counter to a dict for serialization
            counts = json.dumps(counts)
            constant_counts.append(counts)
        except Exception as e:
            logging.debug(f"Error counting constants in code: {code}, error: {e}")
            constant_counts.append({})
    return {'constant_counts': constant_counts}

dataset = dataset.map(
    add_constant_counts,
    batched=True,
    desc="Counting constants in code",
    num_proc=1
)

# Optional: print the constant counts for the first example to verify
print(dataset[0]['constant_counts'])

global_constant_counts = Counter()
for sample in dataset:
    global_constant_counts.update(json.loads(sample['constant_counts']))

# top 20:
print(global_constant_counts.most_common(20))
