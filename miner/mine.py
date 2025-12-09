import hashlib
import ast
import logging
import re
import datetime
import json
import argparse
import copy
import tiktoken
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager, Lock
from sugar_utils import filter_code, get_compound_stmt_head
from datasets import load_from_disk, Dataset, load_dataset
from tree import Tree
from anonymizer import anonymize, fix_wildcard_order
from collections import Counter
from reward import transform_expr, transform_stmt, transform_stmt_head


UNSUPPORTED_EXPR = ["Name", "Constant", "Slice", "Tuple"]
UNSUPPORTED_EXPR = {expr: True for expr in UNSUPPORTED_EXPR}


def _mine(args_tuple):
    code, stmt_counter, stmt_head_counter, expr_counter, file_counter = args_tuple
    ast_tree = ast.parse(code)
    tmp_stmt_file_count = {}
    tmp_expr_file_count = {}
    try:
        tree = anonymize(ast_tree)
    except RecursionError:
        return
    tree = Tree(ast_tree)
    # need a method for compound statement heads
    simple_stmts, compound_stmts = tree.get_all_statements()
    for stmt in compound_stmts:
        stmt_head = get_compound_stmt_head(stmt)
        stmt_head = fix_wildcard_order(stmt_head)
        logging.debug(f"Compound statement head: {stmt_head}")
        stmt_head_counter[stmt_head] = stmt_head_counter.get(stmt_head, 0) + 1
        tmp_stmt_file_count[stmt_head] = 1

    # single statement
    for stmt in simple_stmts:
        # fix bugs / fix large body
        stmt_str = ast.unparse(stmt)
        stmt_str = fix_wildcard_order(stmt_str)
        stmt_counter[stmt_str] = stmt_counter.get(stmt_str, 0) + 1
        tmp_stmt_file_count[stmt_str] = 1
        
    # expression
    for expr in tree.get_all_expressions():
        if type(expr).__name__ in UNSUPPORTED_EXPR:
            continue
        expr_str = ast.unparse(expr)
        expr_str = fix_wildcard_order(expr_str)
        expr_counter[expr_str] = expr_counter.get(expr_str, 0) + 1
        tmp_expr_file_count[expr_str] = 1

    # multiple adjacent statements
    for n in range(2, 8):
        for stmt_group in tree.get_all_adjacent_statements(n):
            group_node = ast.Module(body=copy.deepcopy(stmt_group), type_ignores=[])
            group_str = ast.unparse(group_node)
            group_str = fix_wildcard_order(group_str)
            stmt_counter[group_str] = stmt_counter.get(group_str, 0) + 1
            tmp_stmt_file_count[group_str] = 1
    logging.debug(f"tmp_stmt_file_count: {tmp_stmt_file_count}")
    for stmt_str, count in tmp_stmt_file_count.items():
        file_counter[stmt_str] = file_counter.get(stmt_str, 0) + count
    for expr_str, count in tmp_expr_file_count.items():
        file_counter[expr_str] = file_counter.get(expr_str, 0) + count



def count_freq(dataset, args):
    start_time = time.time()

    if args.use_pool:
        manager = Manager()
        stmt_counter = manager.dict()
        stmt_head_counter = manager.dict()
        expr_counter = manager.dict()
        file_counter = manager.dict()

        tasks = [(code, stmt_counter, stmt_head_counter, expr_counter, file_counter)
                for code in dataset["content"]]

        with Pool(processes=args.num_proc) as pool:
            for _ in tqdm(pool.imap_unordered(_mine, tasks), total=len(tasks)):
                pass

    else:
        stmt_counter = Counter()
        stmt_head_counter = Counter()
        expr_counter = Counter()
        file_counter = Counter()
        for code in tqdm(dataset["content"], desc="Counting frequency"):
            args_tuple = (code, stmt_counter, stmt_head_counter, expr_counter, file_counter)
            _mine(args_tuple)
    stmt_counter = Counter(dict(stmt_counter))
    expr_counter = Counter(dict(expr_counter))
    print(f"Time taken: {time.time() - start_time} seconds")
    return stmt_counter, stmt_head_counter, expr_counter, file_counter



class Miner:
    def __init__(self, args, dataset):
        self.dataset = dataset
        if args.threshold < 1:
            self.threshold = int(len(self.dataset) * args.threshold)
        else:
            self.threshold = args.threshold
        cl100k_base = tiktoken.get_encoding("cl100k_base")
        self.allowed_special = set(["<stmt_token>", "<expr_token>", "<\expr_token>"]) | set(cl100k_base._special_tokens)

        self.tokenizer = tiktoken.Encoding(
            name="cl100k_im",
            pat_str=cl100k_base._pat_str,
            mergeable_ranks=cl100k_base._mergeable_ranks,
            special_tokens={
                **cl100k_base._special_tokens,
                "<stmt_token>": 100264,
                "<expr_token>": 100265,
                "<\expr_token>": 100266,
            }
        )
        self.error_counter = 0

    def run(self, args):
        self.dataset_total_tokens = self._count_tokens(self.dataset)
        logging.info(f"Total tokens: {self.dataset_total_tokens}")
        raw_stmt_counter, raw_stmt_head_counter, raw_expr_counter, self.file_counter = count_freq(self.dataset, args)
        # change the counter to dict
        logging.info(f"Counter before filter: {len(raw_stmt_counter)}, {len(raw_expr_counter)}, {len(raw_stmt_head_counter)}")
 
        logging.info(f"Filter with threshold: {self.threshold}")
        logging.debug(f"stmt_head_counter: {raw_stmt_head_counter}")
        self.stmt_head_counter, self.stmt_counter, self.expr_counter = {}, {}, {}
        for k, v in self.file_counter.items():
            if v >= self.threshold:
                if k in raw_stmt_head_counter:
                    self.stmt_head_counter[k] = raw_stmt_head_counter.get(k, 0)
                elif k in raw_stmt_counter:
                    self.stmt_counter[k] = raw_stmt_counter.get(k, 0)
                elif k in raw_expr_counter:
                    self.expr_counter[k] = raw_expr_counter.get(k, 0)
       
        logging.info(f"After Step 1: {len(self.stmt_counter)} stmt, {len(self.expr_counter)} expr, {len(self.stmt_head_counter)} stmt_head")

        sugar = self._create_sugar(self.stmt_counter, 'stmt')
        sugar.extend(self._create_sugar(self.expr_counter, 'expr')) 
        sugar.extend(self._create_sugar(self.stmt_head_counter, 'stmt_head'))
        print(f'After Step 2: {len(sugar)} left')

        sugar = sorted(sugar, key=lambda x: x['saved'], reverse=True)
        print(f"Total sugar saved: {sum([x['saved'] for x in sugar])}")
        print(f"Total sugar: {len(sugar)}")
        print(f"Total tokens: {self.dataset_total_tokens}")

        with open(f'./results/sugar_{args.dataset_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.json', 'w') as f:
            json.dump({
                'dataset_name': args.dataset_name,
                'num_sugar': len(sugar),
                'sugar': sugar,
            }, f, indent=4)
    
    def _create_sugar(self, counter, count_type='stmt'):
        sugar = []
        for k, v in counter.items():
            try:
                if count_type == 'stmt_head':
                    simplified = transform_stmt_head(k)
                elif count_type == 'stmt':
                    simplified = transform_stmt(k)
                elif count_type == 'expr':
                    simplified = transform_expr(k)
                else:
                    raise ValueError(f"Invalid count_type: {count_type}")
            except Exception as e:
                self.error_counter += 1
                continue
            original_tokens = len(self.tokenizer.encode(k))
            simplified_tokens = len(self.tokenizer.encode(simplified, allowed_special=self.allowed_special))
            reward = original_tokens - simplified_tokens
            if reward > args.min_reward:
                sugar.append({
                    'code': k,
                    'reward': reward,
                    'saved': reward * v,
                    'freq': v,
                    'file_freq': self.file_counter.get(k, 0),
                    'type': count_type,
                    'id': hashlib.md5(k.encode()).hexdigest(),
                })
        return sugar


    def _count_tokens(self, dataset):
        def count_tokens(examples):
            token_nums = [len(self.tokenizer.encode(code, allowed_special=self.allowed_special)) for code in examples['content']]
            examples['token_num'] = token_nums
            return examples
        dataset = dataset.map(count_tokens, batched=True, batch_size=10000, desc="Counting tokens")
        total_tokens = sum(dataset['token_num'])
        return total_tokens




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="/workspace/cached")
    parser.add_argument("--use_pool", action="store_true")
    parser.add_argument("--num_proc", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--min_reward", type=int, default=2)
    parser.add_argument("--log_level", type=str, default="WARNING",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Set the logging level")
    args = parser.parse_args()
    
    # Set log level based on argument
    logging.basicConfig(level=getattr(logging, args.log_level))
    

    dataset = load_dataset("LimYeri/LeetCode_Python_Solutions_v2")['train']
    new_examples = []
    def extract_content(examples):
        new_examples = []
        for content in examples['content']:
            try:
                code = content.split('```python')[1].split('```')[0]
                ast.parse(code)
                new_examples.append(code)
            except Exception as e:
                logging.debug(f"Error parsing content: {content}")
                continue
        return {'content': new_examples}
    dataset = dataset.map(extract_content, batched=True, desc="Extracting content from LeetCode", remove_columns=[c for c in dataset.column_names if c != 'content'], num_proc=args.num_proc)
    logging.info(f"Extracted {len(dataset)} LeetCode solutions") 

    miner = Miner(args, dataset)
    miner.run(args)
    