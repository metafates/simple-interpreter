from __future__ import annotations
import string as _string
from enum import Enum, auto

# <CONSTANTS>
from typing import Callable

LETTERS = _string.ascii_letters
DIGITS = _string.digits

FN_OPERATOR = '=>'
EQ_OPERATOR = '='
PLUS_OPERATOR = '+'
MINUS_OPERATOR = '-'
MUL_OPERATOR = '*'
DIV_OPERATOR = '/'
REM_OPERATOR = '%'

OPERATORS = (FN_OPERATOR, PLUS_OPERATOR, MINUS_OPERATOR,
             MUL_OPERATOR, DIV_OPERATOR, REM_OPERATOR)
OPERATORS_CHARSET = set(''.join(OPERATORS))

FN_KEYWORD = 'fn'
KEYWORDS = (FN_KEYWORD,)


class TokenType(Enum):
    OPERATOR = auto()
    IDENTIFIER = auto()
    NUMBER = auto()
    KEYWORD = auto()
    EOF = auto()
    LEFT_PAR = auto()
    RIGHT_PAR = auto()
# </CONSTANTS>


# <TOKEN>
class Token:
    def __init__(self, token_type: TokenType, value: str = ""):
        self.type = token_type
        self.value = value

    def __str__(self):
        if self.value:
            return f'{self.type.name}: {self.value}'
        return f'{self.type.name}'

    def __repr__(self):
        return self.__str__()
# </TOKEN>


# <LEXER>
class Lexer:
    char: str | None
    pos: int
    stream: str

    def __init__(self, stream: str):
        self.char = None
        self.pos = -1
        self.stream = stream

        self.advance()

    def advance(self) -> None:
        self.pos += 1
        if self.pos < len(self.stream):
            self.char = self.stream[self.pos]
        else:
            self.char = None

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []

        while self.char is not None:
            match self.char:
                case ' ':
                    self.advance()
                case parenthesis if parenthesis in ')(':
                    is_left = parenthesis == '('
                    t = Token(TokenType.LEFT_PAR if is_left else TokenType.RIGHT_PAR)
                    tokens.append(t)
                    self.advance()
                case operator if operator in OPERATORS_CHARSET:
                    operator = self.make_operator()
                    token = Token(TokenType.OPERATOR, operator)
                    tokens.append(token)
                case number if number in DIGITS:
                    number = self.make_number()
                    token = Token(TokenType.NUMBER, number)
                    tokens.append(token)
                case keywordOrId if keywordOrId in LETTERS + '_':
                    maybe_id = self.make_identifier()
                    if maybe_id in KEYWORDS:
                        token = Token(TokenType.KEYWORD, maybe_id)
                    else:
                        token = Token(TokenType.IDENTIFIER, maybe_id)
                    tokens.append(token)
                case _:
                    raise Exception(f'Unknown Symbol {self.char}')
        eof = Token(TokenType.EOF)
        tokens.append(eof)

        return tokens

    def make_operator(self) -> str:
        op = ''

        def looks_like_operator(o) -> bool:
            return any(map(lambda x: x.startswith(o), OPERATORS))

        while self.char is not None and looks_like_operator(op + self.char):
            op += self.char
            self.advance()

        return op

    def make_number(self) -> str:
        num = ''
        dot = False

        while self.char is not None and self.char in DIGITS + '.':
            if self.char == '.':
                if dot:
                    break
                dot = True
            num += self.char
            self.advance()

        return num

    def make_identifier(self) -> str:
        identifier = ''

        while self.char is not None and self.char in LETTERS + '_' + DIGITS:
            identifier += self.char
            self.advance()

        return identifier
# </LEXER>


# <NODES>
class Nodes:
    class Node:
        def __init__(self, token: Token = None):
            self.token = token
            self.ids: set[Nodes.Identifier] = set()

        def add_identifier(self, identifier: Nodes.Node):
            if isinstance(identifier, Nodes.Identifier):
                self.ids.add(identifier)

        def __str__(self):
            return str(self.token)

        def __repr__(self):
            return self.__str__()

    class Number(Node):
        def __init__(self, token: Token):
            super().__init__(token)

    class Identifier(Node):
        def __init__(self, token: Token):
            super().__init__(token)
            self.name = self.token.value

            self.add_identifier(self)

        def __str__(self):
            return self.name

    class BinOp(Node):
        def __init__(self, left_operand: Nodes.Node, operator: Token, right_operand: Nodes.Node):
            super().__init__()
            self.left_operand = left_operand
            self.operator = operator
            self.right_operand = right_operand

            for identifier in left_operand.ids:
                self.add_identifier(identifier)
            self.add_identifier(left_operand)

            for identifier in right_operand.ids:
                self.add_identifier(identifier)
            self.add_identifier(right_operand)

        def __str__(self):
            return f'({self.left_operand}, {self.operator}, {self.right_operand})'

    class UnaryOp(Node):
        def __init__(self, operator: Token, operand: Nodes.Node):
            super().__init__()
            self.operator = operator
            self.operand = operand

            for identifier in operand.ids:
                self.add_identifier(identifier)
            self.add_identifier(operand)

        def __str__(self):
            return f'({self.operator}, {self.operand})'

    class VariableAssignment(Node):
        def __init__(self, variable: Token, expression: Nodes.Node):
            super().__init__()
            self.variable = variable
            self.expression = expression

            self.add_identifier(expression)

        def __str__(self):
            return f'({self.variable} = {self.expression})'

    class FunctionAssignment(Node):
        def __init__(
                self,
                function: Nodes.Identifier,
                arguments: list[Nodes.Identifier],
                expression: Nodes.Node
        ):
            super().__init__()
            self.function = function
            self.arguments = arguments
            self.expression = expression

            self.add_identifier(function)
            self.add_identifier(expression)
            for arg in arguments:
                self.add_identifier(arg)
            self.__check_for_args_scope()

        # That's just bad...
        def __check_for_args_scope(self):
            if len(set(self.arguments)) != len(self.arguments):
                raise Exception('Repeating arguments in function definition')
            if len(self.expression.ids) != len(self.arguments):
                print(self.expression.ids, self.arguments)
                raise Exception('Unknown identifier in function body')

        def __str__(self):
            variables_names = ' '.join(map(lambda v: v.name, self.arguments))
            return f'({FN_KEYWORD} {self.function.name} [{variables_names}] => {self.expression})'

    class FunctionCall(Node):
        def __init__(self, function: Nodes.Identifier, arguments: list[Nodes.Identifier]):
            super().__init__()
            self.function = function
            self.arguments = arguments

            self.add_identifier(function)
            # Not sure about it?
            for arg in arguments:
                self.add_identifier(arg)

        def __str__(self):
            # arguments = ' '.join(map(lambda v: v.name, self.arguments))
            return f'({self.function.name}[{" ".join(map(str, self.arguments))}])'
# </NODES>


# <Parser>
class Parser:
    tokens: list[Token]
    token: Token | None
    pos: int

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = -1
        self.token = None
        self.advance()

    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.token = self.tokens[self.pos]
        else:
            self.token = None

    def generateAST(self) -> Nodes.Node:
        ast = self.expression()
        if self.token.type is not TokenType.EOF:
            raise Exception('Invalid Syntax')
        return ast

    def factor(self) -> Nodes.Node | Nodes.Identifier:
        token = self.token

        match token.type:
            case TokenType.OPERATOR:
                # Unary Operator
                if token.value in (PLUS_OPERATOR, MINUS_OPERATOR):
                    self.advance()
                    factor = self.factor()
                    return Nodes.UnaryOp(token, factor)

                # Variable assign
                elif token.value == EQ_OPERATOR:
                    self.advance()
                    expression = self.expression()
                    return Nodes.VariableAssignment(token, expression)
                elif token.value == FN_OPERATOR:
                    self.advance()
                    return Nodes.Node()

                raise Exception('Invalid Operator')
            case TokenType.NUMBER:
                self.advance()
                return Nodes.Number(token)
            case TokenType.LEFT_PAR:
                self.advance()
                expression = self.expression()
                if self.token.type is TokenType.RIGHT_PAR:
                    self.advance()
                    return expression
            case TokenType.IDENTIFIER:
                self.advance()
                return Nodes.Identifier(token)
            case TokenType.KEYWORD:
                self.advance()
                # Function assignment
                if token.value == FN_KEYWORD:
                    function_identifier = self.factor()
                    variables = []
                    last_return = self.factor()

                    # Collect variables until '=>' operator is met
                    while isinstance(last_return, Nodes.Identifier):
                        variables.append(last_return)
                        last_return = self.factor()

                    expression = self.expression()
                    return Nodes.FunctionAssignment(function_identifier, variables, expression)

                raise Exception('Unknown Keyword')

        raise Exception('Unknown Token Type')

    def term(self) -> Nodes.Node:
        term_operators = (MUL_OPERATOR, DIV_OPERATOR, REM_OPERATOR)
        return self.bin_op_or_function_call(self.factor, term_operators)

    def expression(self) -> Nodes.Node:
        expression_operators = (PLUS_OPERATOR, MINUS_OPERATOR)
        return self.bin_op_or_function_call(self.term, expression_operators)

    def bin_op_or_function_call(
            self,
            fn: Callable[..., Nodes.Node | Nodes.Identifier],
            operators: tuple[str, ...]
    ) -> Nodes.BinOp | Nodes.Node:
        left_operand = fn()

        # This is awful solution lol but whatever (it should be done in the factor)
        if self.token.value == EQ_OPERATOR:
            self.advance()
            expression = self.expression()
            return Nodes.VariableAssignment(left_operand.token, expression)

        # Check if it is function call
        # Example: add echo 4 echo 3 -> 7.
        arguments = []
        while self.token.type in (TokenType.IDENTIFIER, TokenType.NUMBER):
            arguments.append(self.factor())

        if arguments:
            return Nodes.FunctionCall(left_operand, arguments)

        while self.token.value in operators:
            operator = self.token
            self.advance()
            right_operand = fn()
            left_operand = Nodes.BinOp(left_operand, operator, right_operand)

        return left_operand
# </Parser>


# <Objects>
class Objects:

    class Object:
        def __init__(self):
            pass

        def __repr__(self):
            return self.__str__()

    class Number(Object):
        def __init__(self, value: int):
            super().__init__()
            self.value = value

        def __str__(self):
            return str(self.value)

        def __add__(self, other: Objects.Number):
            return Objects.Number(self.value + other.value)

        def __sub__(self, other: Objects.Number):
            return Objects.Number(self.value - other.value)

        def __mul__(self, other: Objects.Number):
            return Objects.Number(self.value * other.value)

        def __floordiv__(self, other: Objects.Number):
            return Objects.Number(self.value // other.value)

        def __mod__(self, other: Objects.Number):
            return Objects.Number(self.value % other.value)

        def __pow__(self, other: Objects.Number, modulo=None):
            return Objects.Number(self.value ** other.value)

    class Variable(Object):
        def __init__(self, name: str, value: Objects.Object):
            super().__init__()
            self.name = name
            self.value = value

        def __str__(self):
            return self.name

    class Function(Object):
        def __init__(self, name: str, variables: list[Nodes.Identifier], body: Nodes.Node):
            super().__init__()
            self.name = name
            self.variables = variables
            self.body = body

        def call(self, arguments: list[Nodes.Identifier]) -> Objects.Object:
            binded_args = {var.name: arguments[i] for i, var in enumerate(self.variables)}
            scope = Interpreter(variables=binded_args)
            return scope.input_ast(self.body)

        def __str__(self):
            return f'{self.name} [{" ".join(map(str, self.variables))}] => {self.body}'
# </Object>


# <Interpreter>
class Interpreter:
    ast: Nodes.Node
    variables: dict[str, Objects.Variable]
    functions: dict[str, Objects.Function]

    def __init__(self, variables=None, functions=None):
        self.variables = variables or dict()
        self.functions = functions or dict()

    def input(self, code: str) -> Objects.Object:
        tokens = Lexer(code).tokenize()
        ast = Parser(tokens).generateAST()
        return self.visit(ast)

    # TODO: remove later
    def input_ast(self, ast: Nodes.Node) -> Objects.Object:
        return self.visit(ast)

    def visit(self, node: Nodes.Node) -> Objects.Object:
        if isinstance(node, Nodes.Number):
            return self.visit_number_node(node)
        if isinstance(node, Nodes.Identifier):
            return self.visit_identifier_node(node)
        if isinstance(node, Nodes.BinOp):
            return self.visit_binary_op_node(node)
        if isinstance(node, Nodes.UnaryOp):
            return self.visit_unary_op_node(node)
        if isinstance(node, Nodes.FunctionCall):
            return self.visit_function_call_node(node)
        if isinstance(node, Nodes.VariableAssignment):
            return self.visit_var_assign_node(node)
        if isinstance(node, Nodes.FunctionAssignment):
            return self.visit_func_assign_node(node)

        raise Exception('Unknown Node')


    def visit_number_node(self, node: Nodes.Number) -> Objects.Number:
        return Objects.Number(int(node.token.value))

    def visit_identifier_node(self, node: Nodes.Identifier) -> Objects.Object:
        name = node.name
        if name in self.variables:
            return self.variables[name]

        if name in self.functions:
            return self.functions[name]
        raise Exception('Unknown identifier')

    def visit_var_assign_node(self, node: Nodes.VariableAssignment) -> Objects.Object:
        variable_name = node.variable.value
        if variable_name in self.functions:
            raise Exception('Cannot reassign function to variable (Name conflict)')
        variable_value = self.visit(node.expression)
        variable = Objects.Variable(variable_name, variable_value)
        self.variables[variable.name] = variable
        return variable.value

    def visit_func_assign_node(self, node: Nodes.FunctionAssignment) -> Objects.Function:
        function_name = node.function.name
        if function_name in self.variables:
            raise Exception('Cannot reassign variable to function (Name conflict)')
        function_vars = node.arguments
        function_body = node.expression

        function = Objects.Function(function_name, function_vars, function_body)
        self.functions[function.name] = function
        return function

    def visit_function_call_node(self, node: Nodes.FunctionCall) -> Objects.Object:
        function = self.functions[node.function.name]
        arguments = node.arguments

        # TODO: implement function call
        return function.call(arguments)

    def visit_unary_op_node(self, node: Nodes.UnaryOp) -> Objects.Number:
        num = self.visit(node.operand)
        if not isinstance(num, Objects.Number):
            raise Exception('Unary operation can be applied only to numbers')

        if node.operator.type is MINUS_OPERATOR:
            num *= Objects.Number(-1)

        return num

    def visit_binary_op_node(self, node: Nodes.BinOp) -> Objects.Number:
        left = self.visit(node.left_operand)
        right = self.visit(node.right_operand)

        print(type(left), type(right))

        if isinstance(left, Nodes.Number):
            left = Objects.Number(int(left.token.value))
        if isinstance(right, Nodes.Number):
            right = Objects.Number(int(right.token.value))

        if not isinstance(left, Objects.Number) or not isinstance(right, Objects.Number):
            raise Exception('Binary operation can be applied only to numbers')

        op = node.operator.value
        if op == PLUS_OPERATOR:
            return left + right
        if op == MINUS_OPERATOR:
            return left - right
        if op == MUL_OPERATOR:
            return left * right
        if op == DIV_OPERATOR:
            if right.value == 0:
                raise Exception('Division by zero')
            return left // right
        if op == REM_OPERATOR:
            if right.value == 0:
                raise Exception('Division by zero')
            return left % right

        raise Exception('Unknown operator')
# </Interpreter>

interpreter = Interpreter()
interpreter.input("fn echo x => x")
interpreter.input("fn add x y => x + y")
var = interpreter.input("add echo 4 echo 3")
print(var)

