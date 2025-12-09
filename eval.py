import time
import json
import csv
from io import StringIO
import ast
import re
import logging

from textwrap import dedent
from modifier.pattern import Pattern, apply_patterns
from rope.refactor.similarfinder import CodeTemplate

# Compile the provided token patterns
match_stmt_token = re.compile(r'<\d+_stmt_token>')
match_expr_start_token = re.compile(r'<\d+_expr_token>')
match_expr_end_token = re.compile(r'<\\expr_token>')
string_literal_pattern = re.compile(
    r"""
    " (?:\\"|\\[^]|[^"\\])* "    
    |                             
    ' (?:\\'|\\[^]|[^'\\])* '   
    |                             
    \"\"\" [\s\S]*? \"\"\"
    |                             
    ''' [\s\S]*? '''
    """, re.VERBOSE
)

combined_token_pattern = re.compile(
    r'(<[a-z0-9]+_stmt_token>|<[a-z0-9]+_expr_token>|<\\expr_token>)'
)

def _safe_split_lines(code):
    lines = []
    buffer = ""
    stack = []
    in_string = False
    string_char = ''
    in_multiline_string = False
    in_comment = False

    def is_opening(c):
        return c in '([{'

    def is_closing(c):
        return c in ')]}'

    def matches(opening, closing):
        return (opening == '(' and closing == ')') or \
               (opening == '[' and closing == ']') or \
               (opening == '{' and closing == '}')

    i = 0
    n = len(code)

    while i < n:
        char = code[i]
        next_char = code[i+1] if i+1 < n else ''

        if in_comment:
            buffer += char
            if char == '\n':
                in_comment = False
                lines.append(buffer)
                buffer = ""
            i += 1
            continue

        if in_string:
            buffer += char
            if char == '\\':
                buffer += next_char
                i += 2
                continue
            if not in_multiline_string and char == string_char:
                in_string = False
            elif in_multiline_string and code[i:i+3] == string_char * 3:
                buffer += code[i+1:i+3]
                in_string = False
                in_multiline_string = False
                i += 3
                continue
            i += 1
            continue

        if char == '#':
            in_comment = True
            buffer += char
            i += 1
            continue

        if char in ('"', "'"):
            if code[i:i+3] == char * 3:
                in_string = True
                in_multiline_string = True
                string_char = char
                buffer += code[i:i+3]
                i += 3
                continue
            else:
                in_string = True
                in_multiline_string = False
                string_char = char
                buffer += char
                i += 1
                continue

        if is_opening(char):
            stack.append(char)
        elif is_closing(char):
            if stack and matches(stack[-1], char):
                stack.pop()
            else:
                stack.clear()

        buffer += char

        if char == '\n':
            if not stack:
                lines.append(buffer)
                buffer = ""

        i += 1

    if buffer:
        lines.append(buffer)

    return lines


def _find_tokens(code):
    lines = []

    code_lines = _safe_split_lines(code)

    for line in code_lines:
        tokens = []
        token_detected = False

        string_literals = []
        for match in string_literal_pattern.finditer(line):
            string_literals.append((match.start(), match.end()))

        def is_inside_string(pos):
            for start, end in string_literals:
                if start <= pos < end:
                    return True
            return False

        for match in combined_token_pattern.finditer(line):
            start, end = match.span()
            if not is_inside_string(start):
                tokens.append(match.group())
                token_detected = True

        lines.append({
            'content': line.rstrip('\n') + '\n',
            'tokens': tokens,
            'token_detected': token_detected
        })

    return lines


def split_vars(code):
    if code.strip() == '':
        return []
    split = [c.strip() for c in code.split(';')]
    return [c for c in split if len(c) > 0]

class Parser:
    def __init__(self):
        self.match_expr = re.compile(r'<[a-z0-9]+_expr_token>')
        self.match_stmt = re.compile(r'<[a-z0-9]+_stmt_token>')
    
    
    def set_patterns(self, pattern_filename):
        patterns = json.load(open(pattern_filename))
        patterns = patterns['sugar']
        self.complete_pattern_list = [Pattern.from_dict(p) for p in patterns if p['type'] != 'stmt_head']
        self.single_expr_pattern_list = [p for p in self.complete_pattern_list if p.pattern_type == 'expr' and '<\expr_token>' not in p.goal]
        self.partial_pattern_list = [Pattern.from_dict(p) for p in patterns if p['type'] == 'stmt_head']

        self.patterns = {f'<{p["id"]}_stmt_token>' if 'stmt' in p['type'] else f'<{p["id"]}_expr_token>': Pattern.from_dict(p) for p in patterns}
    
    def is_single_stmt(self, token):
        for p in self.complete_pattern_list + self.single_expr_pattern_list:
            if p.goal == token:
                return True
        return False


    def is_partial_stmt(self, token):
        for p in self.partial_pattern_list:
            if f'<{p.id}_stmt_token>' == token:
                return True
        return False

    def is_single_expr(self, token):
        for p in self.single_expr_pattern_list:
            if f'<{p.id}_expr_token>' == token:
                return True
        return False
    

    def encode(self, code):
        encoded_code = apply_patterns({'content': [code]}, self.complete_pattern_list, self.partial_pattern_list)['content'][0]
        return encoded_code


    def parse(self, code):
        parse_record = []
        lines = _find_tokens(code)
        converted_lines = []
        logging.info(f'lines: {lines}')
        for line in lines:
            stmt_token = ''
            expr_tokens = []
            for token in line['tokens']:
                logging.debug(f'expr match: {self.match_expr.match(token)}, token: {token}')
                if self.match_expr.match(token):
                    expr_tokens.append(token)
                elif self.match_stmt.match(token):
                    stmt_token = token
            logging.info(f'stmt_token: {stmt_token}, expr_tokens: {expr_tokens}')
            if not expr_tokens and not stmt_token:
                converted_lines.append(line['content'])
                continue
            
            if expr_tokens and stmt_token:
                expr_parsed = self._parse_expr(line['content'], expr_tokens)
                stmt_parsed = self._parse_stmt(expr_parsed, stmt_token)
                parsed = stmt_parsed
            elif stmt_token:
                parsed = self._parse_stmt(line['content'], stmt_token)
            elif expr_tokens:
                parsed = self._parse_expr(line['content'], expr_tokens)
            converted_lines.append(parsed)
            parse_record.append({
                'tokens': [stmt_token] + expr_tokens,
                'pre_code': line['content'],
                'post_code': parsed
            })
        parsed = ''.join(converted_lines)
        return parsed, parse_record

    def _parse_expr(self, code, expr_tokens):
        for token in expr_tokens:
            logging.debug(f'Processing token: {token}')
            logging.debug(f'Pattern: {self.patterns[token]}')
            start_index = code.find(token) + len(token)
            if self.is_single_expr(token):
                end_index = -1
            else:
                end_index = code.find('<\expr_token>', start_index)
            logging.debug(f'Start index: {start_index}, End index: {end_index}')
            if end_index == -1:
                if 'SUGARWILDCARD_0' not in self.patterns[token].pattern:
                    applied_expr = self.patterns[token].pattern
                    code = code[:start_index - len(token)] + applied_expr + code[start_index:]
                else:
                    logging.debug('End index not found, continuing to next token')
                    continue
            else:
                vars =code[start_index:end_index]
                logging.debug(f'Extracted vars: {vars}')
                vars = split_vars(vars)
                logging.debug(f'Split vars: {vars}')
                applied_expr = self._apply_pattern(token, vars, is_stmt=False)
                logging.debug(f'Applied expression: {applied_expr}')
                if applied_expr is None:
                    logging.debug('Applied expression is None, continuing to next token')
                    continue
                code = code[:start_index - len(token)] + applied_expr + code[end_index+len('<\expr_token>'):]
            logging.debug(f'Updated code: {code}')
        return code

    def _parse_stmt(self, code, stmt_token):
        logging.debug(f"Parsing statement with code: {code} and stmt_token: {stmt_token}")
        logging.debug(f"Is single stmt: {self.is_single_stmt(stmt_token)}")
        logging.debug(f"Pattern is: {self.patterns[stmt_token].pattern}")
        if self.is_single_stmt(stmt_token):
            return code.replace(stmt_token, self.patterns[stmt_token].pattern)
        indents = re.match(r'^(\s*)', code).group(0)
        logging.debug(f"Detected indents: '{indents}'")
        
        expected_written_vars, expected_read_vars = self.patterns[stmt_token].goal.split(stmt_token)
        logging.debug(f"Expected written vars: {expected_written_vars}, Expected read vars: {expected_read_vars}")
        
        written_vars_str, read_vars_str = code.split(stmt_token)
        logging.debug(f"Written vars string: {written_vars_str}, Read vars string: {read_vars_str}")
        
        written_vars = split_vars(written_vars_str)
        expected_written_vars = split_vars(expected_written_vars)
        logging.debug(f"Split written vars: {written_vars}, Split expected written vars: {expected_written_vars}")
        
        if len(written_vars) != len(expected_written_vars):
            written_vars = written_vars_str.split(',')
    
        read_vars = split_vars(read_vars_str)
        expected_read_vars = split_vars(expected_read_vars)
        logging.debug(f"Split read vars: {read_vars}, Split expected read vars: {expected_read_vars}")
        
        if len(read_vars) != len(expected_read_vars):
            logging.debug("Mismatch in number of read vars, splitting by ','")
            read_vars = read_vars_str.split(',')

        vars = written_vars + read_vars
        logging.debug(f"Combined vars: {vars}")
                
        if stmt_token not in self.patterns:
            logging.error(f"Statement token {stmt_token} not found in patterns")
            raise ValueError(f"Statement token {stmt_token} not found in patterns")
        
        
        applied_stmt = self._apply_pattern(stmt_token, vars)
        logging.debug(f"Applied statement: {applied_stmt}")
        
        split_applied_stmt = [indents+s for s in applied_stmt.split('\n')]
        result = '\n'.join(split_applied_stmt) + '\n' if applied_stmt is not None else code
        logging.debug(f"Final parsed statement: {result}")
        
        return result
    

    def _apply_pattern(self, token, vars, is_stmt=True):
        mapping = {}
        goal = self.patterns[token].goal
        logging.debug(f'goal: {goal}')
        if len(vars) == 0:
            return self.patterns[token].pattern 
        if is_stmt:
            processed_goal = goal.replace(token, ';').strip(';')
            expected_vars = processed_goal.split(';')
            
            if len(expected_vars) != len(vars):
                return None
            logging.debug(f'expected_vars: {expected_vars}, vars: {vars}')
            max_index = -1
            for wildcard in expected_vars:
                wildcard = wildcard.strip()[2:-1]
                max_index = max(max_index, int(wildcard.split('_')[-1]))
                mapping[wildcard] = vars.pop(0)
            if self.patterns[token].pattern_type == 'stmt_head':
                # find the largest index of the wildcard in the goal
                mapping[f'SUGARWILDCARD_{max_index+1}'] = ''
        else:
            expected_vars = goal[len(token):-len('<\expr_token>')].split(';')
            if len(expected_vars) != len(vars):
                return None
            for wildcard in expected_vars:
                wildcard = wildcard.strip()[2:-1]
                mapping[wildcard] = vars.pop(0)
        
        original_pattern = CodeTemplate(self.patterns[token].pattern)  
        logging.debug(f'mapping: {mapping}, original_pattern: {self.patterns[token].pattern}, type: {self.patterns[token].pattern_type}')
        return original_pattern.substitute(mapping) if self.patterns[token].pattern_type != 'stmt_head' else original_pattern.substitute(mapping).strip()

class Converter:
    def __init__(self, pattern_filename):
        self.patterns = [Pattern.from_dict(p) for p in json.load(open(pattern_filename))]

    def convert(self, sugar_code):
        return self.patterns[0].apply(sugar_code)
