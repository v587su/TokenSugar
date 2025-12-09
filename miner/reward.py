import ast
import unittest
from collections import OrderedDict
from contextlib import contextmanager


class VariableCollector(ast.NodeVisitor):
    '''Find all names starting with "SUGARWILDCARD_" in the AST'''

    def __init__(self):
        self.names = []

    def _check_and_add(self, name):
        if name.startswith("SUGARWILDCARD_"):
            self.names.append(name)

    def _process_arguments(self, arguments):
        for arg in arguments.posonlyargs:
            self._check_and_add(arg.arg)
        for arg in arguments.args:
            self._check_and_add(arg.arg)
        if arguments.vararg:
            self._check_and_add(arguments.vararg.arg)
        for arg in arguments.kwonlyargs:
            self._check_and_add(arg.arg)
        if arguments.kwarg:
            self._check_and_add(arguments.kwarg.arg)

    def visit_Name(self, node):
        self._check_and_add(node.id)

    def visit_FunctionDef(self, node):
        self._check_and_add(node.name)
        self._process_arguments(node.args)
        self.generic_visit(node) 

    def visit_AsyncFunctionDef(self, node):
        self._check_and_add(node.name)
        self._process_arguments(node.args)
        self.generic_visit(node)

    def visit_Lambda(self, node):
        self._process_arguments(node.args)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self._check_and_add(node.name)
        self.generic_visit(node)
        

def _get_wildcard_names(node):
    collector = VariableCollector()
    collector.visit(node)
    return collector.names


def transform_expr(expr_str):
    # <start_token>WILDCARD_0, WILDCARD_1, ...<end_token>
    tree = ast.parse(expr_str)
    names = _get_wildcard_names(tree)
    if len(names) == 0:
        return '<expr_token>'
    else:   
        # Sort names to ensure consistent ordering
        names.sort(key=lambda x: int(x.split('_')[1]))
        new_expr_str = '<expr_token>' + ';'.join(names) + '<\expr_token>'
        return new_expr_str


def transform_stmt(stmt_str):
    # Find the written variables in the statement. 
    # written variables are before <stmt_token>, read variables are after <stmt_token>
    # e.g., WILDCARD_0 = WILDCARD_1 + 1 -> WILDCARD_0<stmt_token>WILDCARD_1
    tree = ast.parse(stmt_str)
    names = _get_wildcard_names(tree)
    if len(names) == 0:
        return '<stmt_token>'
    collector = WrittenVariableCollector()
    collector.visit(tree)
    written_vars = set(collector.written)
    read_vars = set(names) - written_vars
    return ';'.join(sorted(list(written_vars))) + '<stmt_token>' + ';'.join(sorted(list(read_vars)))
    # return '<stmt_token>' + ';'.join(sorted(names))

def transform_stmt_head(stmt_head_str):
    complete_stmt_str = stmt_head_str + '\n    pass'
    return transform_stmt(complete_stmt_str)


class WrittenVariableCollector(ast.NodeVisitor):
    def __init__(self):
        self.written = set()
        self.conditional = False
        self.loop_depth = 0

    def collect(self, node):
        self.visit(node)
        return self.written.copy()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.written.add(node.id)

    def visit_FunctionDef(self, node):
        # both function name and parameters are written variables
        self.written.add(node.name)
        for arg in node.args.args:
            self.written.add(arg.arg)

    def visit_AsyncFunctionDef(self, node):
        self.written.add(node.name)

    def visit_ClassDef(self, node):
        self.written.add(node.name)

    def visit_For(self, node):
        with self._handle_loop_context():
            self.visit(node.target)
            self.visit(node.iter)
            for stmt in node.body:
                self.visit(stmt)
            for stmt in node.orelse:
                self.visit(stmt)

    def visit_While(self, node):
        with self._handle_loop_context():
            self.visit(node.test)
            for stmt in node.body:
                self.visit(stmt)
            for stmt in node.orelse:
                self.visit(stmt)

    def visit_If(self, node):
        with self._handle_conditional_context():
            self.visit(node.test)
            for stmt in node.body:
                self.visit(stmt)
            for stmt in node.orelse:
                self.visit(stmt)

    def visit_Assign(self, node):
        for target in node.targets:
            self.visit(target)
        self.visit(node.value)

    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            self.written.add(node.target.id)
        self.visit(node.value)

    def visit_AnnAssign(self, node):
        self.visit(node.target)
        if node.value:
            self.visit(node.value)

    def visit_With(self, node):
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars:
                self.visit(item.optional_vars)
        for stmt in node.body:
            self.visit(stmt)

    def visit_Import(self, node):
        for alias in node.names:
            self.written.add(alias.asname or alias.name)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.written.add(alias.asname or alias.name)

    def visit_Module(self, node):
        for stmt in node.body:
            self.visit(stmt)

    def _get_argnames(self, args):
        for arg in args.args + args.kwonlyargs:
            yield arg.arg
        if args.vararg:
            yield args.vararg.arg
        if args.kwarg:
            yield args.kwarg.arg

    @contextmanager
    def _handle_conditional_context(self):
        prev_conditional = self.conditional
        self.conditional = True
        try:
            yield
        finally:
            self.conditional = prev_conditional

    @contextmanager
    def _handle_loop_context(self):
        self.loop_depth += 1
        try:
            yield
        finally:
            self.loop_depth -= 1

# Unit tests
class TestWrittenVariableCollector(unittest.TestCase):
    def test_assign(self):
        code = """
x = 5
y = x + 3
"""
        parsed_code = ast.parse(code)
        collector = WrittenVariableCollector()
        written_vars = collector.collect(parsed_code)
        self.assertEqual(written_vars, {'x', 'y'})

    def test_aug_assign(self):
        code = """
x = 1
x += 2
"""
        parsed_code = ast.parse(code)
        collector = WrittenVariableCollector()
        written_vars = collector.collect(parsed_code)
        self.assertEqual(written_vars, {'x'})

    def test_function_def(self):
        code = """
def foo(a, b):
    c = a + b
"""
        parsed_code = ast.parse(code)
        collector = WrittenVariableCollector()
        written_vars = collector.collect(parsed_code)
        self.assertEqual(written_vars, {'foo'})

    def test_class_def(self):
        code = """
class MyClass:
    x = 10
    def method(self):
        pass
"""
        parsed_code = ast.parse(code)
        collector = WrittenVariableCollector()
        written_vars = collector.collect(parsed_code)
        self.assertEqual(written_vars, {'MyClass'})

    def test_import(self):
        code = """
import os
import numpy as np
"""
        parsed_code = ast.parse(code)
        collector = WrittenVariableCollector()
        written_vars = collector.collect(parsed_code)
        self.assertEqual(written_vars, {'os', 'np'})
