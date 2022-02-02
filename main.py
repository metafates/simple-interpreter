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


a = Lexer("fn doubleIt x => x * 2")
print(a.tokenize())
