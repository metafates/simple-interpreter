from __future__ import annotations
import string as _string
from enum import Enum, auto

# <CONSTANTS>
LETTERS = _string.ascii_letters
DIGITS = _string.digits

FN_OPERATOR = '=>'
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
        def __init__(self, token: Token):
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
        def __init__(self, left: Nodes.Node, op: Token, right: Nodes.Node):
            super().__init__(op)
            self.left = left
            self.right = right

        def __str__(self):
            return f'({self.left}, {self.token}, {self.right})'

    class VariableAssignment(Node):
        def __init__(self, variable: Token, expression: Nodes.Node):
            super().__init__(variable)
            self.expression = expression

        def __str__(self):
            return f'({self.token} = {self.expression})'

    class FunctionAssignment(Node):
        def __init__(
                self,
                function: Token,
                variables: list[Nodes.Identifier],
                expression: Nodes.Node
        ):
            super().__init__(function)
            self.variables = variables
            self.expression = expression

        def __str__(self):
            variables_names = ' '.join(map(lambda v: v.name, self.variables))
            return f'({self.token.value} [{variables_names}] = {self.expression})'

    class FunctionCall(Node):
        def __init__(self, function: Token, variables: list[Nodes.Identifier]):
            super().__init__(function)
            self.variables = variables

        def __str__(self):
            variables_names = ' '.join(map(lambda v: v.name, self.variables))
            return f'({self.token.value}[{variables_names}])'
# </NODES>


a = Lexer("fn doubleIt x => x * 2")
tokens = a.tokenize()
b = Nodes.FunctionAssignment(tokens[1], [Nodes.Identifier(tokens[2])], Nodes.BinOp(Nodes.Number(tokens[4]), tokens[5], Nodes.Identifier(tokens[6])))
print(b)
