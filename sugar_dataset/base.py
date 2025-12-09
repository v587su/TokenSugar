import time
import shutil
import re
import os
import tiktoken
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from modifier.pattern import Pattern, apply_patterns
from modifier.modify import _MyPatchingASTWalker

from rope.base import ast

from datasets import load_from_disk, load_dataset


class BaseDataset:
    def __init__(self, args):
        self.args = args
        self.pattern_filename = args.pattern_file_path
        self.patterns = self.load_patterns() 
        self.special_tokens = self._get_special_tokens()
        self.dataset = self.load_dataset()


        self.dataset = self.filter_dataset(self.dataset)
        # check the self name
        self.dataset = self.dataset.map(self._create_response, batched=True, desc="Adding prefix and postfix")


    def load_patterns(self):
        patterns = json.load(open(self.pattern_filename))
        if patterns.get('sugar'):
            patterns = patterns['sugar']
        print(f"Loaded {len(patterns)} patterns")
        return [Pattern.from_dict(p) for p in patterns]
    
    
    def get_token_num(self, tokenizer):
        def count_old_tokens(examples):
            token_nums = [len(tokenizer(code)) for code in examples['old_content']]
            examples['old_token_num'] = token_nums
            return examples

        def count_tokens(examples):
            token_nums = [len(tokenizer(code)) for code in examples['content']]
            examples['token_num'] = token_nums
            return examples
       
        dataset = self.dataset.map(count_tokens, batched=True, batch_size=10000, desc="Counting tokens", num_proc=self.args.num_proc)
        dataset = dataset.map(count_old_tokens, batched=True, batch_size=10000, desc="Counting old tokens", num_proc=self.args.num_proc)
        total_tokens = sum(dataset['token_num'])
        total_old_tokens = sum(dataset['old_token_num'])
        print(f"Total tokens: {total_tokens}, Total old tokens: {total_old_tokens}")
        return total_tokens, total_old_tokens
    
    def _get_special_tokens(self):
        if not self.args.is_sugarized:
            return []
        special_tokens = set()
        for p in self.patterns:
            special_tokens = special_tokens | p.special_tokens
        return sorted(list(special_tokens))

    def load_dataset(self):
        if self.args.load_from == 'hub':
            assert self.args.dataset_name is not None, "dataset_name is required when loading from hub"
            dataset = load_dataset(self.args.dataset_name, split='train')
        elif self.args.load_from == 'new':
           
            dataset = self._load()
            if self.args.is_sugarized:
                dataset = self.sugarize(dataset)
        else:
            raise ValueError(f"Invalid load_from: {self.args.load_from}")
        return dataset
    
    def _load(self):
        re_first_line = re.compile(r'^.*\n')
        hf_home = os.environ.get('HF_HOME')
        dataset = load_dataset(self.args.dataset_name, split="train", cache_dir=hf_home)
        def filter_content(examples):
            new_examples = []
            for sample in examples['content']:
                try:
                    first_line = re_first_line.match(sample).group()
                except:
                    continue

                if first_line.startswith('<'):
                    sample = re_first_line.sub('', sample) 

                if sample.startswith('#!/'):
                    continue
                
                try:
                    ast.parse(sample)
                except Exception as e:
                    continue
                
                new_examples.append(sample)
            return {'content': new_examples}
      
        dataset = dataset.map(filter_content, batched=True, desc="Filtering content", num_proc=self.args.num_proc, remove_columns=list(set(dataset.column_names) - {'content'}))

        return dataset

    def sugarize(self, dataset):
        dataset = dataset.map(apply_patterns, batched=True, fn_kwargs={'complete_patterns':[p for p in self.patterns if p.pattern_type != 'stmt_head'], 'partial_patterns':[p for p in self.patterns if p.pattern_type == 'stmt_head']}, desc="Applying patterns", num_proc=self.args.num_proc, load_from_cache_file=False)
        return dataset

