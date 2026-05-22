"""
Mini Compiler - CSE 430 Compiler Design Lab
Student ID: 22101184
University of Asia Pacific

Phases:
1. Lexical Analysis (Lexer)
2. Syntax Analysis (Parser)
3. Semantic Analysis
4. Intermediate Code Generation (Three-Address Code)
5. Code Generation (Assembly-like output)
"""

import re
import sys

# ─────────────────────────────────────────────
# PHASE 1: LEXER (Lexical Analysis)
# ─────────────────────────────────────────────

TOKEN_SPEC = [
    ('NUMBER',   r'\d+(\.\d*)?'),
    ('STRING',   r'"[^"]*"'),
    ('IF',       r'\bif\b'),
    ('ELSE',     r'\belse\b'),
    ('WHILE',    r'\bwhile\b'),
    ('INT',      r'\bint\b'),
    ('FLOAT',    r'\bfloat\b'),
    ('RETURN',   r'\breturn\b'),
    ('PRINT',    r'\bprint\b'),
    ('DEF',      r'\bdef\b'),
    ('ID',       r'[A-Za-z_]\w*'),
    ('ASSIGN',   r'=='),
    ('NEQ',      r'!='),
    ('LEQ',      r'<='),
    ('GEQ',      r'>='),
    ('EQ',       r'='),
    ('LT',       r'<'),
    ('GT',       r'>'),
    ('PLUS',     r'\+'),
    ('MINUS',    r'-'),
    ('TIMES',    r'\*'),
    ('DIVIDE',   r'/'),
    ('LPAREN',   r'\('),
    ('RPAREN',   r'\)'),
    ('LBRACE',   r'\{'),
    ('RBRACE',   r'\}'),
    ('SEMICOLON',r';'),
    ('COMMA',    r','),
    ('COLON',    r':'),
    ('NEWLINE',  r'\n'),
    ('SKIP',     r'[ \t]+'),
    ('COMMENT',  r'//[^\n]*'),
    ('MISMATCH', r'.'),
]

Token = lambda type, value, line: {'type': type, 'value': value, 'line': line}

def lexer(code):
    tokens = []
    line_num = 1
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPEC)
    for mo in re.finditer(tok_regex, code):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'NEWLINE':
            line_num += 1
            continue
        elif kind in ('SKIP', 'COMMENT'):
            continue
        elif kind == 'MISMATCH':
            raise SyntaxError(f'[Lexer Error] Unexpected character {value!r} at line {line_num}')
        else:
            tokens.append(Token(kind, value, line_num))
    return tokens


# ─────────────────────────────────────────────
# PHASE 2: PARSER (Syntax Analysis) - Recursive Descent
# ─────────────────────────────────────────────

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def current(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token('EOF', '', -1)

    def consume(self, expected_type=None):
        tok = self.current()
        if expected_type and tok['type'] != expected_type:
            raise SyntaxError(
                f"[Parser Error] Expected {expected_type} but got {tok['type']} ('{tok['value']}') at line {tok['line']}"
            )
        self.pos += 1
        return tok

    def parse(self):
        stmts = []
        while self.current()['type'] != 'EOF':
            stmts.append(self.parse_statement())
        return ('PROGRAM', stmts)

    def parse_statement(self):
        tok = self.current()

        if tok['type'] in ('INT', 'FLOAT'):
            return self.parse_declaration()
        elif tok['type'] == 'IF':
            return self.parse_if()
        elif tok['type'] == 'WHILE':
            return self.parse_while()
        elif tok['type'] == 'DEF':
            return self.parse_function()
        elif tok['type'] == 'RETURN':
            return self.parse_return()
        elif tok['type'] == 'PRINT':
            return self.parse_print()
        elif tok['type'] == 'ID':
            return self.parse_assign_or_call()
        elif tok['type'] == 'RBRACE':
            return None
        else:
            raise SyntaxError(f"[Parser Error] Unexpected token {tok['type']} at line {tok['line']}")

    def parse_declaration(self):
        dtype = self.consume()['value']
        name = self.consume('ID')['value']
        self.consume('EQ')
        expr = self.parse_expression()
        self.consume('SEMICOLON')
        return ('DECL', dtype, name, expr)

    def parse_assign_or_call(self):
        name = self.consume('ID')['value']
        if self.current()['type'] == 'EQ':
            self.consume('EQ')
            expr = self.parse_expression()
            self.consume('SEMICOLON')
            return ('ASSIGN', name, expr)
        elif self.current()['type'] == 'LPAREN':
            args = self.parse_args()
            self.consume('SEMICOLON')
            return ('CALL', name, args)
        else:
            raise SyntaxError(f"[Parser Error] Expected = or ( after identifier '{name}'")

    def parse_if(self):
        self.consume('IF')
        self.consume('LPAREN')
        cond = self.parse_condition()
        self.consume('RPAREN')
        self.consume('LBRACE')
        then_body = self.parse_block()
        self.consume('RBRACE')
        else_body = []
        if self.current()['type'] == 'ELSE':
            self.consume('ELSE')
            self.consume('LBRACE')
            else_body = self.parse_block()
            self.consume('RBRACE')
        return ('IF', cond, then_body, else_body)

    def parse_while(self):
        self.consume('WHILE')
        self.consume('LPAREN')
        cond = self.parse_condition()
        self.consume('RPAREN')
        self.consume('LBRACE')
        body = self.parse_block()
        self.consume('RBRACE')
        return ('WHILE', cond, body)

    def parse_function(self):
        self.consume('DEF')
        name = self.consume('ID')['value']
        self.consume('LPAREN')
        params = []
        while self.current()['type'] != 'RPAREN':
            params.append(self.consume('ID')['value'])
            if self.current()['type'] == 'COMMA':
                self.consume('COMMA')
        self.consume('RPAREN')
        self.consume('LBRACE')
        body = self.parse_block()
        self.consume('RBRACE')
        return ('FUNC', name, params, body)

    def parse_return(self):
        self.consume('RETURN')
        expr = self.parse_expression()
        self.consume('SEMICOLON')
        return ('RETURN', expr)

    def parse_print(self):
        self.consume('PRINT')
        self.consume('LPAREN')
        expr = self.parse_expression()
        self.consume('RPAREN')
        self.consume('SEMICOLON')
        return ('PRINT', expr)

    def parse_block(self):
        stmts = []
        while self.current()['type'] not in ('RBRACE', 'EOF'):
            s = self.parse_statement()
            if s: stmts.append(s)
        return stmts

    def parse_args(self):
        self.consume('LPAREN')
        args = []
        while self.current()['type'] != 'RPAREN':
            args.append(self.parse_expression())
            if self.current()['type'] == 'COMMA':
                self.consume('COMMA')
        self.consume('RPAREN')
        return args

    def parse_condition(self):
        left = self.parse_expression()
        op_types = ('ASSIGN','NEQ','LT','GT','LEQ','GEQ')
        if self.current()['type'] in op_types:
            op = self.consume()['value']
            right = self.parse_expression()
            return ('COND', op, left, right)
        return left

    def parse_expression(self):
        node = self.parse_term()
        while self.current()['type'] in ('PLUS', 'MINUS'):
            op = self.consume()['value']
            right = self.parse_term()
            node = ('BINOP', op, node, right)
        return node

    def parse_term(self):
        node = self.parse_factor()
        while self.current()['type'] in ('TIMES', 'DIVIDE'):
            op = self.consume()['value']
            right = self.parse_factor()
            node = ('BINOP', op, node, right)
        return node

    def parse_factor(self):
        tok = self.current()
        if tok['type'] == 'NUMBER':
            self.consume()
            return ('NUM', tok['value'])
        elif tok['type'] == 'STRING':
            self.consume()
            return ('STR', tok['value'])
        elif tok['type'] == 'ID':
            self.consume()
            if self.current()['type'] == 'LPAREN':
                args = self.parse_args()
                return ('CALL', tok['value'], args)
            return ('VAR', tok['value'])
        elif tok['type'] == 'LPAREN':
            self.consume('LPAREN')
            node = self.parse_expression()
            self.consume('RPAREN')
            return node
        elif tok['type'] == 'MINUS':
            self.consume()
            factor = self.parse_factor()
            return ('UNARY', '-', factor)
        else:
            raise SyntaxError(f"[Parser Error] Unexpected token {tok['type']} ('{tok['value']}') at line {tok['line']}")


# ─────────────────────────────────────────────
# PHASE 3: SEMANTIC ANALYSIS
# ─────────────────────────────────────────────

class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = {}
        self.functions = {}
        self.errors = []

    def analyze(self, node):
        if node[0] == 'PROGRAM':
            for stmt in node[1]:
                self.analyze(stmt)
        elif node[0] == 'DECL':
            _, dtype, name, expr = node
            if name in self.symbol_table:
                self.errors.append(f"[Semantic Error] Variable '{name}' already declared.")
            self.symbol_table[name] = dtype
            self.analyze(expr)
        elif node[0] == 'ASSIGN':
            _, name, expr = node
            if name not in self.symbol_table:
                self.errors.append(f"[Semantic Warning] Variable '{name}' used before declaration.")
                self.symbol_table[name] = 'int'
            self.analyze(expr)
        elif node[0] == 'IF':
            self.analyze(node[1])
            for s in node[2]: self.analyze(s)
            for s in node[3]: self.analyze(s)
        elif node[0] == 'WHILE':
            self.analyze(node[1])
            for s in node[2]: self.analyze(s)
        elif node[0] == 'FUNC':
            _, name, params, body = node
            self.functions[name] = params
            for p in params:
                self.symbol_table[p] = 'int'
            for s in body: self.analyze(s)
        elif node[0] == 'RETURN':
            self.analyze(node[1])
        elif node[0] == 'PRINT':
            self.analyze(node[1])
        elif node[0] == 'BINOP':
            self.analyze(node[2])
            self.analyze(node[3])
        elif node[0] == 'COND':
            self.analyze(node[2])
            self.analyze(node[3])
        elif node[0] == 'VAR':
            if node[1] not in self.symbol_table:
                self.errors.append(f"[Semantic Error] Variable '{node[1]}' not declared.")
        elif node[0] == 'CALL':
            for arg in node[2]: self.analyze(arg)

    def print_symbol_table(self):
        print("\n--- Symbol Table ---")
        for name, dtype in self.symbol_table.items():
            print(f"  {name:<15} : {dtype}")
        for fn, params in self.functions.items():
            print(f"  {fn:<15} : function({', '.join(params)})")


# ─────────────────────────────────────────────
# PHASE 4: INTERMEDIATE CODE GENERATION (3-Address Code)
# ─────────────────────────────────────────────

class ICGGenerator:
    def __init__(self):
        self.code = []
        self.temp_count = 0
        self.label_count = 0

    def new_temp(self):
        self.temp_count += 1
        return f't{self.temp_count}'

    def new_label(self):
        self.label_count += 1
        return f'L{self.label_count}'

    def emit(self, instr):
        self.code.append(instr)

    def generate(self, node):
        if node[0] == 'PROGRAM':
            for stmt in node[1]:
                self.generate(stmt)

        elif node[0] == 'DECL':
            _, dtype, name, expr = node
            t = self.gen_expr(expr)
            self.emit(f'{name} = {t}')

        elif node[0] == 'ASSIGN':
            _, name, expr = node
            t = self.gen_expr(expr)
            self.emit(f'{name} = {t}')

        elif node[0] == 'IF':
            _, cond, then_body, else_body = node
            false_label = self.new_label()
            end_label = self.new_label()
            self.gen_cond(cond, false_label)
            for s in then_body: self.generate(s)
            self.emit(f'goto {end_label}')
            self.emit(f'{false_label}:')
            for s in else_body: self.generate(s)
            self.emit(f'{end_label}:')

        elif node[0] == 'WHILE':
            _, cond, body = node
            start_label = self.new_label()
            end_label = self.new_label()
            self.emit(f'{start_label}:')
            self.gen_cond(cond, end_label)
            for s in body: self.generate(s)
            self.emit(f'goto {start_label}')
            self.emit(f'{end_label}:')

        elif node[0] == 'FUNC':
            _, name, params, body = node
            self.emit(f'func {name}:')
            for p in params:
                self.emit(f'param {p}')
            for s in body: self.generate(s)
            self.emit(f'end func {name}')

        elif node[0] == 'RETURN':
            t = self.gen_expr(node[1])
            self.emit(f'return {t}')

        elif node[0] == 'PRINT':
            t = self.gen_expr(node[1])
            self.emit(f'print {t}')

        elif node[0] == 'CALL':
            _, name, args = node
            for arg in args:
                t = self.gen_expr(arg)
                self.emit(f'push {t}')
            self.emit(f'call {name}')

    def gen_expr(self, node):
        if node[0] == 'NUM':
            return node[1]
        elif node[0] == 'STR':
            return node[1]
        elif node[0] == 'VAR':
            return node[1]
        elif node[0] == 'UNARY':
            t1 = self.gen_expr(node[2])
            t = self.new_temp()
            self.emit(f'{t} = -{t1}')
            return t
        elif node[0] == 'BINOP':
            t1 = self.gen_expr(node[2])
            t2 = self.gen_expr(node[3])
            t = self.new_temp()
            self.emit(f'{t} = {t1} {node[1]} {t2}')
            return t
        elif node[0] == 'CALL':
            _, name, args = node
            for arg in args:
                ta = self.gen_expr(arg)
                self.emit(f'push {ta}')
            self.emit(f'call {name}')
            t = self.new_temp()
            self.emit(f'{t} = retval')
            return t
        return '?'

    def gen_cond(self, node, false_label):
        if node[0] == 'COND':
            t1 = self.gen_expr(node[2])
            t2 = self.gen_expr(node[3])
            inv = {'==':'!=','!=':'==','<':'>=','>':'<=','<=':'>','>=':'<'}.get(node[1], node[1])
            self.emit(f'if {t1} {inv} {t2} goto {false_label}')
        else:
            t = self.gen_expr(node)
            self.emit(f'if {t} == 0 goto {false_label}')

    def print_code(self):
        print("\n--- Intermediate Code (Three-Address Code) ---")
        for i, line in enumerate(self.code):
            print(f"  {i+1:3}: {line}")


# ─────────────────────────────────────────────
# PHASE 5: CODE GENERATION (Assembly-like)
# ─────────────────────────────────────────────

class CodeGenerator:
    def __init__(self, ic_code):
        self.ic = ic_code
        self.asm = []

    def generate(self):
        self.asm.append("; === Generated Assembly Code ===")
        self.asm.append(".data")
        self.asm.append(".code")
        self.asm.append("START:")

        for line in self.ic:
            line = line.strip()
            if not line:
                continue

            # Label
            if line.endswith(':'):
                self.asm.append(f"\n{line}")

            # goto
            elif line.startswith('goto '):
                label = line.split()[1]
                self.asm.append(f"    JMP {label}")

            # conditional if a op b goto L
            elif line.startswith('if '):
                parts = line.split()
                # if t1 op t2 goto Label
                t1, op, t2, _, lbl = parts[1], parts[2], parts[3], parts[4], parts[5]
                op_map = {'!=':'JNE','==':'JE','<':'JL','>':'JG','<=':'JLE','>=':'JGE'}
                asm_op = op_map.get(op, 'JNE')
                self.asm.append(f"    MOV AX, {t1}")
                self.asm.append(f"    CMP AX, {t2}")
                self.asm.append(f"    {asm_op} {lbl}")

            # assignment: x = a op b
            elif '=' in line and not line.startswith('func') and not line.startswith('end'):
                lhs, rhs = line.split('=', 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                tokens = rhs.split()
                if len(tokens) == 3:
                    a, op, b = tokens
                    op_map = {'+':'ADD','-':'SUB','*':'MUL','/':'DIV'}
                    asm_op = op_map.get(op, 'ADD')
                    self.asm.append(f"    MOV AX, {a}")
                    self.asm.append(f"    {asm_op} AX, {b}")
                    self.asm.append(f"    MOV {lhs}, AX")
                elif len(tokens) == 2 and tokens[0] == '-':
                    self.asm.append(f"    MOV AX, {tokens[1]}")
                    self.asm.append(f"    NEG AX")
                    self.asm.append(f"    MOV {lhs}, AX")
                else:
                    val = tokens[0] if tokens else '0'
                    self.asm.append(f"    MOV AX, {val}")
                    self.asm.append(f"    MOV {lhs}, AX")

            # print
            elif line.startswith('print '):
                val = line.split(' ', 1)[1]
                self.asm.append(f"    MOV AX, {val}")
                self.asm.append(f"    OUT AX")

            # return
            elif line.startswith('return '):
                val = line.split(' ', 1)[1]
                self.asm.append(f"    MOV AX, {val}")
                self.asm.append(f"    RET")

            # push
            elif line.startswith('push '):
                val = line.split(' ', 1)[1]
                self.asm.append(f"    PUSH {val}")

            # call
            elif line.startswith('call '):
                fn = line.split(' ', 1)[1]
                self.asm.append(f"    CALL {fn}")

            # func
            elif line.startswith('func '):
                fn = line.split(' ', 1)[1].rstrip(':')
                self.asm.append(f"\n{fn} PROC")

            elif line.startswith('end func'):
                fn = line.split()[-1]
                self.asm.append(f"    RET")
                self.asm.append(f"{fn} ENDP")

        self.asm.append("\nHLT")
        return self.asm

    def print_code(self):
        print("\n--- Generated Assembly Code ---")
        for line in self.asm:
            print(f"  {line}")


# ─────────────────────────────────────────────
# MAIN DRIVER
# ─────────────────────────────────────────────

SAMPLE_CODE = """
int x = 5;
int y = 10;
int z = x + y;

if (z > 10) {
    print(z);
} else {
    print(x);
}

int i = 0;
while (i < 5) {
    i = i + 1;
    print(i);
}

def add(a, b) {
    int result = a + b;
    return result;
}
"""

def compile_code(source):
    print("=" * 55)
    print("   MINI COMPILER  |  Student ID: 22101184")
    print("=" * 55)

    # Phase 1
    print("\n[Phase 1] Lexical Analysis...")
    try:
        tokens = lexer(source)
        print(f"  Tokens generated: {len(tokens)}")
        print("  Token list (first 15):")
        for t in tokens[:15]:
            print(f"    [{t['type']:12}] {t['value']}")
        if len(tokens) > 15:
            print(f"    ... and {len(tokens)-15} more tokens")
    except SyntaxError as e:
        print(e); return

    # Phase 2
    print("\n[Phase 2] Syntax Analysis (Parsing)...")
    try:
        parser = Parser(tokens)
        ast = parser.parse()
        print(f"  AST root: {ast[0]}, Statements: {len(ast[1])}")
    except SyntaxError as e:
        print(e); return

    # Phase 3
    print("\n[Phase 3] Semantic Analysis...")
    sa = SemanticAnalyzer()
    sa.analyze(ast)
    sa.print_symbol_table()
    if sa.errors:
        print("\n  Errors/Warnings:")
        for e in sa.errors:
            print(f"  ⚠  {e}")
    else:
        print("  No semantic errors found.")

    # Phase 4
    print("\n[Phase 4] Intermediate Code Generation...")
    icg = ICGGenerator()
    icg.generate(ast)
    icg.print_code()

    # Phase 5
    print("\n[Phase 5] Code Generation (Assembly)...")
    cg = CodeGenerator(icg.code)
    cg.generate()
    cg.print_code()

    print("\n" + "=" * 55)
    print("  Compilation Successful!")
    print("=" * 55)

    return icg.code, cg.asm

if __name__ == '__main__':
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            source = f.read()
    else:
        source = SAMPLE_CODE
    compile_code(source)
