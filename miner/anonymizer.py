import ast
import logging
import time
from collections import defaultdict
import itertools
import copy
import re
from textwrap import dedent
# disable logging
PREFIX = "_SUGARWILDCARD_"


class ParentSetter(ast.NodeTransformer):
    def __init__(self):
        self.node_id = 0

    def visit(self, node):
        node.node_id = self.node_id
        self.node_id += 1
        for child in ast.iter_child_nodes(node):
            child.parent = node  # Set the parent attribute for child nodes
            self.visit(child)
        return node


def _get_parent_code(node):
    code_list = []
    if not hasattr(node, 'parent'):
        return code_list
    parent = node.parent
    if not isinstance(parent, ast.expr):
        return code_list
    parent_code = ast.unparse(parent)
    # check whether multiple different wildcards are in the parent code
    # we don't keep parent of multiple wildcards
    wildcards = set([match.group(0) for match in re.finditer(PREFIX + r'\d+', parent_code)])
    if len(wildcards) > 1:
        return code_list
    code_list.append(parent_code)
    code_list.extend(_get_parent_code(parent))
    return code_list


def _get_variants(code, parent_map):
    non_empty_items = {k: v for k, v in parent_map.items() if v}
    keys = list(non_empty_items.keys())
    if len(keys) == 0:
        return [code]
    highest_number = max([int(k.split('_')[2]) for k in parent_map.keys()])
    value_lists = [v + [None] for v in non_empty_items.values()]
    all_combinations = list(itertools.product(*value_lists))
    selected_combos = [
        {k: v for k, v in zip(keys, combination) if v is not None}
        for combination in all_combinations
    ]
    variants = []
    for selected_combo in selected_combos:
        code_to_replace = code
        counter = highest_number
        for k, v in selected_combo.items():
            if v is None:
                continue
            # count the number of occurences of v and k in code_to_replace
            need_new_name = code_to_replace.count(v) != code_to_replace.count(k)
            if need_new_name:
                # highest number in the wildcard name + 1
                new_name = f"{PREFIX}{counter + 1}"
                counter += 1
            else:
                new_name = k
            code_to_replace = code_to_replace.replace(v, new_name)

        variants.append(code_to_replace)
    return variants


class VariableAbstractor(ast.NodeTransformer):
    def __init__(self, with_order=True, max_counter=10000):
        self.name_map = {}
        self.const_map = {}
        self.counter = -1
        self.const_counter = -1
        self.max_counter = max_counter
        self.with_order = with_order
        # Allowed constants that won't be abstracted
        self.allowed_constants = {0, 1, -1, 2, 3, 10, True, False, None, ""}

    def get_name(self, node_id):
        if node_id in self.name_map:
            return self.name_map[node_id]
        self.counter += 1
        if self.counter > self.max_counter:
            raise ValueError("Variable counter exceeded max_counter")
        name = f"{PREFIX}{self.counter}" if self.with_order else f"{PREFIX}0"
        self.name_map[node_id] = name
        return name

    def visit_Name(self, node):
        parent = getattr(node, 'parent', None)
        if parent is not None:
            if isinstance(parent, ast.Attribute) and parent.attr == node.id:
                return node
            if isinstance(parent, ast.Call) and parent.func is node:
                return node
        
        if isinstance(node.ctx, (ast.Store, ast.Load)):
            name = self.get_name(node.id)
            new_node = ast.copy_location(ast.Name(id=name, ctx=node.ctx), node)
            new_node.parent = node.parent
            new_node.is_wildcard = True
            new_node.node_id = node.node_id
            return new_node
        return node

    def visit_Constant(self, node):
        if node.value in self.allowed_constants:
            return node
        parent = getattr(node, 'parent', None)
        if parent and isinstance(parent, ast.JoinedStr):
            return node
        const_name = self.get_name(node.value)
        return ast.copy_location(ast.Name(id=const_name, ctx=ast.Load()), node)

    # Compatibility for older Python versions
    def visit_Num(self, node):
        return self.visit_Constant(node)

    def visit_Str(self, node):
        return self.visit_Constant(node)

    def visit_NameConstant(self, node):
        return self.visit_Constant(node)

    def visit_FunctionDef(self, node):
        # Abstract the function name
        node.name = self.get_name(node.name)

        # Abstract the parameters
        for arg in node.args.args:
            arg.arg = self.get_name(arg.arg)

        # Visit the body of the function to abstract variables inside it
        self.generic_visit(node)
        return node

class ParentFinder(ast.NodeVisitor):
    def __init__(self):
        self.parent_code_map = defaultdict(list)

    def visit(self, node):
        if hasattr(node, 'is_wildcard') and node.is_wildcard:
            parent_code = _get_parent_code(node)
            self.parent_code_map[node.id].extend(parent_code)
        self.generic_visit(node)


reg = re.compile(PREFIX + r'\d+')
def anonymize(tree, with_order=True, max_counter=None):
    # Set parent attributes for all nodes
    # start_time = time.time()
    ParentSetter().visit(tree)
    # Annotate the tree with depth information
    transformer = VariableAbstractor(with_order=with_order)
    try:
        tree = transformer.visit(tree)
    except ValueError as e:
        return []

    return tree



def fix_wildcard_order(code):
    # Extract wildcard names in the order they appear
    wildcard_names = []
    seen = set()
    for match in re.finditer(PREFIX + r'\d+', code):
        wildcard = match.group(0)
        if wildcard not in seen:
            wildcard_names.append(wildcard)
            seen.add(wildcard)


    if len(wildcard_names) == 0:
        return code

    # Create a mapping based on the order of appearance
    wildcard_map = {wildcard: f"{PREFIX[1:]}{i}" for i, wildcard in enumerate(wildcard_names)}

    # replace large wildcard names first
    wildcard_map = sorted(wildcard_map.items(), key=lambda x: len(x[0]), reverse=True)
    for wildcard, new_name in wildcard_map:
        code = code.replace(wildcard, new_name)

    return code


