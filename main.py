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

        def __str__(self):
            return self.name

    class BinOp(Node):
        def __init__(self, left_operand: Nodes.Node, operator: Token, right_operand: Nodes.Node):
            super().__init__()
            self.left_operand = left_operand
            self.operator = operator
            self.right_operand = right_operand

        def __str__(self):
            return f'({self.left_operand}, {self.operator}, {self.right_operand})'

    class UnaryOp(Node):
        def __init__(self, operator: Token, operand: Nodes.Node):
            super().__init__()
            self.operator = operator
            self.operand = operand

        def __str__(self):
            return f'({self.operator}, {self.operand})'

    class VariableAssignment(Node):
        def __init__(self, variable: Token, expression: Nodes.Node):
            super().__init__()
            self.variable = variable
            self.expression = expression

        def __str__(self):
            return f'({self.variable} = {self.expression})'

    class FunctionAssignment(Node):
        def __init__(
                self,
                function: Nodes.Identifier,
                variables: list[Nodes.Identifier],
                expression: Nodes.Node
        ):
            super().__init__()
            self.function = function
            self.variables = variables
            self.expression = expression

        def __str__(self):
            variables_names = ' '.join(map(lambda v: v.name, self.variables))
            return f'({self.function.name} [{variables_names}] = {self.expression})'

    class FunctionCall(Node):
        def __init__(self, function: Token, variables: list[Nodes.Identifier]):
            super().__init__()
            self.function = function
            self.variables = variables

        def __str__(self):
            variables_names = ' '.join(map(lambda v: v.name, self.variables))
            return f'({self.token.value}[{variables_names}])'
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
        return self.bin_op(self.factor, term_operators)

    def expression(self) -> Nodes.Node:
        expression_operators = (PLUS_OPERATOR, MINUS_OPERATOR)
        return self.bin_op(self.term, expression_operators)

    def bin_op(
            self,
            fn: Callable[..., Nodes.Node],
            operators: tuple[str, ...]
    ) -> Nodes.BinOp:
        left_operand = fn()

        while self.token.value in operators:
            operator = self.token
            self.advance()
            right_operand = fn()
            left_operand = Nodes.BinOp(left_operand, operator, right_operand)

        return left_operand
# </Parser>

a = Lexer("(1 + 2) * 3")
tokens = a.tokenize()

b = Parser(tokens)
print(b.generateAST())
