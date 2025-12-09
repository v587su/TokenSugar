import re
import copy
import ast

re_first_line = re.compile(r'^.*\n')
def filter_code(examples):
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
            tree = ast.parse(sample)
        except:
            continue
        new_examples.append(sample)
    return {'content': new_examples}


def get_compound_stmt_head(stmt):
    stmt_copy = copy.deepcopy(stmt)
    stmt_copy.body = [ast.Pass()]
    stmt_str = ast.unparse(stmt_copy)
    stmt_str = stmt_str.replace('pass', '').strip()
    return stmt_str
