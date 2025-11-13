# -*- coding: utf-8 -*-
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Tuple
import json
import sys
import os

class TokenType(Enum):
    IDENT = auto()
    NUMBER = auto()
    BOOLEAN = auto()
    STRING = auto()
    KEYWORD = auto()
    OP = auto()
    SEMICOLON = auto()
    LBRACE = auto()
    RBRACE = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int
    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, L{self.line},C{self.col})"


class Lexer:
    KEYWORDS = {"IF","THEN","END","SET","MOVE","STOP","AND","OR","NOT"}
    BOOLEAN_WORDS = {"true","false","True","False","TRUE","FALSE"}

    def __init__(self, src: str):
        self.src = src
        self.i = 0
        self.line = 1
        self.col = 1

    def peek(self):
        return self.src[self.i] if self.i < len(self.src) else "\0"

    def advance(self):
        ch = self.peek()
        self.i += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def skip(self):
        while True:
            ch = self.peek()
            if ch == "\0": return
            if ch.isspace():
                self.advance(); continue
            if ch == '#':
                while self.peek() != "\0" and self.peek() != "\n":
                    self.advance()
                continue
            break

    def read_ident(self):
        start_col = self.col
        s = ""
        while True:
            ch = self.peek()
            if ch == "\0": break
            if ch.isalnum() or ch == "_":
                s += self.advance()
            else: break
        if s in Lexer.BOOLEAN_WORDS:
            return Token(TokenType.BOOLEAN, s.lower(), self.line, start_col)
        up = s.upper()
        if up in Lexer.KEYWORDS:
            return Token(TokenType.KEYWORD, up, self.line, start_col)
        return Token(TokenType.IDENT, s, self.line, start_col)

    def read_number(self):
        start_col = self.col
        s = ""
        dot = 0
        while True:
            ch = self.peek()
            if ch == ".":
                dot += 1
                if dot > 1: break
                s += self.advance()
            elif ch.isdigit():
                s += self.advance()
            else: break
        return Token(TokenType.NUMBER, s, self.line, start_col)

    def read_string(self):
        start_col = self.col
        self.advance()  # consume "
        s = ""
        while True:
            ch = self.peek()
            if ch == "\0":
                raise Exception(f"String não finalizada L{self.line}")
            if ch == '"':
                self.advance(); break
            if ch == "\\":
                self.advance()
                nxt = self.peek()
                if nxt == "n": s += "\n"; self.advance()
                elif nxt == "t": s += "\t"; self.advance()
                elif nxt == '"': s += '"'; self.advance()
                elif nxt == "\\": s += "\\"; self.advance()
                else: s += self.advance()
                continue
            s += self.advance()
        return Token(TokenType.STRING, s, self.line, start_col)

    def next_token(self):
        self.skip()
        ch = self.peek()
        if ch == "\0":
            return Token(TokenType.EOF, "", self.line, self.col)
        if ch.isalpha() or ch == "_":
            return self.read_ident()
        if ch.isdigit():
            return self.read_number()
        if ch == '"':
            return self.read_string()
        if ch in "{}();":
            start_col = self.col
            self.advance()
            mapping = {'{':TokenType.LBRACE,'}':TokenType.RBRACE,'(':TokenType.LPAREN,')':TokenType.RPAREN,';':TokenType.SEMICOLON}
            return Token(mapping[ch], ch, self.line, start_col)
        if ch in "=!><+-*/":
            start_col = self.col
            a = self.advance()
            if a in ("=","!","<",">") and self.peek() == "=":
                b = self.advance()
                return Token(TokenType.OP, a+b, self.line, start_col)
            return Token(TokenType.OP, a, self.line, start_col)
        raise Exception(f"Caractere inválido '{ch}' na linha {self.line}")

    def tokenize(self):
        toks = []
        while True:
            t = self.next_token()
            toks.append(t)
            if t.type == TokenType.EOF: break
        return toks


@dataclass
class Node: pass
@dataclass
class Program(Node):
    stmts: List[Node]
@dataclass
class Block(Node):
    stmts: List[Node]
@dataclass
class IfNode(Node):
    cond: Node
    then_blk: Block
@dataclass
class SetNode(Node):
    name: str
    expr: Node
@dataclass
class CommandNode(Node):
    name: str  # MOVE / STOP
@dataclass
class Compare(Node):
    left: Node
    op: str
    right: Node
@dataclass
class Logical(Node):
    left: Node
    op: str  # AND / OR
    right: Node
@dataclass
class NumberNode(Node):
    value: float
@dataclass
class BoolNode(Node):
    value: bool
@dataclass
class StringNode(Node):
    value: str
@dataclass
class VarNode(Node):
    name: str
@dataclass
class SemaforoAction(Node):
    sem: str
    action: str
@dataclass
class RoutePriority(Node):
    priority: str
    rota: str
    condition: Node


class ParseError(Exception): pass

class Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.pos = 0
        self.semaforo_ids = {"sem_quadra", "sem_av_principal", "sem_hospital"}
        self.actions = {"abrir", "fechar", "intermitente"}
        self.rotas = {"rota_emergencial", "rota_escolar", "rota_publica"}
        self.priorities = {"alta","media","baixa"}
    def peek(self): return self.toks[self.pos] if self.pos < len(self.toks) else Token(TokenType.EOF,"",-1,-1)
    def next(self): t=self.peek(); self.pos+=1; return t
    def match(self, ttype, value=None):
        tk = self.peek()
        if tk.type==ttype and (value is None or tk.value==value):
            return self.next()
        return None
    def expect(self, ttype, value=None):
        tk = self.match(ttype,value)
        if not tk: raise ParseError(f"Esperado {ttype.name} {value if value else ''}, encontrado {self.peek()}")
        return tk
    def parse(self):
        stmts=[]
        while self.peek().type!=TokenType.EOF:
            if self.peek().type==TokenType.SEMICOLON:
                self.next(); continue
            stmts.append(self.statement())
        return Program(stmts)
    def statement(self):
        tk = self.peek()
        
        if tk.type==TokenType.KEYWORD and tk.value=="SET":
            self.next()
            name_tok = self.expect(TokenType.IDENT)
            self.expect(TokenType.OP,"=")
            expr = self.expr_or()
            self.match(TokenType.SEMICOLON)
            return SetNode(name_tok.value, expr)
        
        if tk.type==TokenType.KEYWORD and tk.value=="IF":
            self.next()
            cond = self.expr_or()
            
            if self.peek().type==TokenType.LBRACE:
                self.next()
                stmts=[]
                while self.peek().type not in (TokenType.RBRACE, TokenType.EOF):
                    if self.peek().type==TokenType.SEMICOLON:
                        self.next(); continue
                    stmts.append(self.statement())
                self.expect(TokenType.RBRACE)
                blk = Block(stmts)
            else:
                stmt = self.statement()
                blk = Block([stmt])
            return IfNode(cond, blk)
        
        if tk.type in (TokenType.IDENT, TokenType.KEYWORD) and tk.value in self.semaforo_ids:
            sem = self.next().value
            act_tok = self.peek()
            if act_tok.type in (TokenType.IDENT, TokenType.KEYWORD) and act_tok.value in self.actions:
                act = self.next().value
                self.match(TokenType.SEMICOLON)
                return SemaforoAction(sem, act)
        
        if tk.type in (TokenType.IDENT, TokenType.KEYWORD) and tk.value in self.priorities:
            pr = self.next().value
            rtok = self.peek()
            if rtok.type in (TokenType.IDENT, TokenType.KEYWORD) and rtok.value in self.rotas:
                rota = self.next().value
                cond = self.expr_or()
                self.match(TokenType.SEMICOLON)
                return RoutePriority(pr, rota, cond)
            else:
                raise ParseError(f"Esperado rota após prioridade, encontrado {rtok}")
        if tk.type==TokenType.KEYWORD and tk.value in ("MOVE","STOP"):
            self.next()
            self.match(TokenType.SEMICOLON)
            return CommandNode(tk.value)
        raise ParseError(f"Comando inválido: {tk}")
    def expr_or(self):
        node = self.expr_and()
        while self.peek().type==TokenType.KEYWORD and self.peek().value=="OR":
            self.next()
            right = self.expr_and()
            node = Logical(node,"OR",right)
        return node
    def expr_and(self):
        node = self.equality()
        while self.peek().type==TokenType.KEYWORD and self.peek().value=="AND":
            self.next()
            right = self.equality()
            node = Logical(node,"AND",right)
        return node
    def equality(self):
        node = self.relation()
        while self.peek().type==TokenType.OP and self.peek().value in ("==","!="):
            op=self.next().value
            right=self.relation()
            node = Compare(node,op,right)
        return node
    def relation(self):
        node = self.term()
        while self.peek().type==TokenType.OP and self.peek().value in ("<",">","<=",">="):
            op=self.next().value
            right=self.term()
            node = Compare(node,op,right)
        return node
    def term(self):
        tk=self.peek()
        if tk.type==TokenType.NUMBER:
            self.next()
            val = float(tk.value) if '.' in tk.value else int(tk.value)
            return NumberNode(val)
        if tk.type==TokenType.BOOLEAN:
            self.next()
            return BoolNode(tk.value.lower()=="true")
        if tk.type==TokenType.STRING:
            self.next()
            return StringNode(tk.value)
        if tk.type==TokenType.IDENT:
            self.next()
            return VarNode(tk.value)
        if tk.type==TokenType.LPAREN:
            self.next()
            node = self.expr_or()
            self.expect(TokenType.RPAREN)
            return node
        raise ParseError(f"Expressão inválida: {tk}")

@dataclass
class SymbolInfo:
    name: str
    kind: str
    type: Optional[str] = None
    value: Any = None

class SymbolTable:
    def __init__(self):
        self.table: Dict[str, SymbolInfo] = {}
    def define(self, name: str, kind: str, typ: Optional[str]=None, value: Any=None):
        self.table[name] = SymbolInfo(name,kind,typ,value)
    def exists(self, name: str) -> bool:
        return name in self.table
    def get(self, name: str) -> Optional[SymbolInfo]:
        return self.table.get(name)
    def items(self):
        return self.table.items()


class SemanticAnalyzer:
    def __init__(self, sensor_env: Optional[Dict[str,Any]] = None):
        self.semaforos = {"sem_quadra","sem_av_principal","sem_hospital"}
        self.rotas = {"rota_emergencial","rota_escolar","rota_publica"}
        self.priorities = {"alta","media","baixa"}
        self.actions = {"abrir","fechar","intermitente"}
        self.errors: List[str] = []
        self.symtab = SymbolTable()
        self.sensor_env = sensor_env or {}
        for k,v in self.sensor_env.items():
            t = self.infer_type(v)
            self.symtab.define(k,"sensor",t,v)

        for s in self.semaforos:
            self.symtab.define(s,"semaforo","string", None)
        for r in self.rotas:
            self.symtab.define(r,"rota","string", None)

    def infer_type(self, value) -> str:
        if isinstance(value, bool): return "bool"
        if isinstance(value, (int,float)): return "number"
        return "string"

    def analyze(self, ast: Program) -> List[str]:
        self.errors = []
        self.visit(ast)
        return self.errors

    def visit(self, node: Node):
        if isinstance(node, Program):
            for s in node.stmts: self.visit(s)
        elif isinstance(node, SetNode):
            t = self.eval_type(node.expr)
            if t is None:
                t = "unknown"
            self.symtab.define(node.name,"var",t,None)
        elif isinstance(node, IfNode):
            t = self.eval_type(node.cond)
            if t is None:
                self.errors.append(f"Condição do IF inválida ou de tipos inconsistentes: {node.cond}")
            self.visit(node.then_blk)
        elif isinstance(node, Block):
            for s in node.stmts: self.visit(s)
        elif isinstance(node, CommandNode):
            if node.name not in ("MOVE","STOP"):
                self.errors.append(f"Comando desconhecido: {node.name}")
        elif isinstance(node, SemaforoAction):
            if node.sem not in self.semaforos:
                self.errors.append(f"Semáforo desconhecido: {node.sem} (sugestões: {sorted(self.semaforos)})")
            if node.action not in self.actions:
                self.errors.append(f"Ação de semáforo inválida: {node.action} (esperado: abrir/fechar/intermitente)")
        elif isinstance(node, RoutePriority):
            if node.priority not in self.priorities:
                self.errors.append(f"Prioridade inválida: {node.priority}")
            if node.rota not in self.rotas:
                self.errors.append(f"Rota inválida: {node.rota}")
            t = self.eval_type(node.condition)
            if t != "bool":
                self.errors.append(f"Condição para rota deve ser booleana (comparação), encontrada: {t}")
        else:
            pass

    def eval_type(self, expr: Node) -> Optional[str]:
        if isinstance(expr, NumberNode): return "number"
        if isinstance(expr, BoolNode): return "bool"
        if isinstance(expr, StringNode): return "string"
        if isinstance(expr, VarNode):
            info = self.symtab.get(expr.name)
            if info: return info.type
            self.errors.append(f"Variável não declarada/nem sensor: {expr.name} (sugestão: definir com SET ou incluir no JSON de sensores)")
            return None
        if isinstance(expr, Compare):
            lt = self.eval_type(expr.left)
            rt = self.eval_type(expr.right)
            if expr.op in ("==","!="):
                if lt is None or rt is None:
                    return None
                if lt != rt:
                    self.errors.append(f"Comparação '==' entre tipos diferentes: {lt} vs {rt}")
                return "bool"
            else:
                if lt == "number" and rt == "number":
                    return "bool"
                self.errors.append(f"Operador '{expr.op}' exige números. Encontrado: {lt} e {rt}")
                return None
        if isinstance(expr, Logical):
            lt = self.eval_type(expr.left)
            rt = self.eval_type(expr.right)
            if lt is None or rt is None:
                return None
            # allow comparisons (which return bool)
            if lt == "bool" and rt == "bool":
                return "bool"
            self.errors.append(f"Operador lógico {expr.op} requer expressões booleanas; obtido {lt} e {rt}")
            return None
        return None


class IRGenerator:
    def generate(self, ast: Program) -> List[Dict[str,Any]]:
        ir = []
        for s in ast.stmts:
            ir.append(self.visit(s))
        return ir
    def visit(self, node: Node):
        if isinstance(node, SetNode):
            return {"op":"set","name":node.name,"expr":self.expr(node.expr)}
        if isinstance(node, CommandNode):
            return {"op":"command","name":node.name}
        if isinstance(node, SemaforoAction):
            return {"op":"semaforo_action","semaforo":node.sem,"action":node.action}
        if isinstance(node, RoutePriority):
            return {"op":"route_priority","priority":node.priority,"rota":node.rota,"condition":self.expr(node.condition)}
        if isinstance(node, IfNode):
            return {"op":"if","condition": self.expr(node.cond), "body":[self.visit(s) for s in node.then_blk.stmts]}
        return {"op":"noop"}
    def expr(self, node: Node):
        if isinstance(node, NumberNode): return {"type":"number","value":node.value}
        if isinstance(node, BoolNode): return {"type":"bool","value":node.value}
        if isinstance(node, StringNode): return {"type":"string","value":node.value}
        if isinstance(node, VarNode): return {"type":"var","name":node.name}
        if isinstance(node, Compare): return {"type":"cmp","op":node.op,"left":self.expr(node.left),"right":self.expr(node.right)}
        if isinstance(node, Logical): return {"type":"logical","op":node.op,"left":self.expr(node.left),"right":self.expr(node.right)}
        return {"type":"unknown"}

class Simulator:
    def __init__(self, ir: List[Dict[str,Any]], sensor_env: Optional[Dict[str,Any]]=None):
        self.ir = ir
        self.env = sensor_env or {}
        self.vars: Dict[str,Any] = {}
        self.semaphores: Dict[str,str] = {}
        self.routes: Dict[str,str] = {}
        self.log: List[str] = []
    def eval_expr(self, expr: Dict[str,Any]):
        t = expr.get("type")
        if t == "number": return expr["value"]
        if t == "bool": return expr["value"]
        if t == "string": return expr["value"]
        if t == "var":
            name = expr["name"]
            if name in self.vars: return self.vars[name]
            return self.env.get(name, False)
        if t == "cmp":
            l = self.eval_expr(expr["left"]); r = self.eval_expr(expr["right"]); op=expr["op"]
            try:
                if op=="==": return l==r
                if op=="!=": return l!=r
                if op=="<": return l<r
                if op==">": return l>r
                if op=="<=": return l<=r
                if op==">=": return l>=r
            except Exception:
                if op in ("==","!="):
                    return l==r if op=="==" else l!=r
                return False
        if t == "logical":
            op = expr["op"]
            if op == "AND":
                return bool(self.eval_expr(expr["left"])) and bool(self.eval_expr(expr["right"]))
            else:
                return bool(self.eval_expr(expr["left"])) or bool(self.eval_expr(expr["right"]))
        return False
    def exec_stmt(self, instr: Dict[str,Any]):
        op = instr.get("op")
        if op=="set":
            val = self.eval_expr(instr["expr"])
            self.vars[instr["name"]] = val
            self.log.append(f"SET {instr['name']} = {val!r}")
        elif op=="command":
            self.log.append(f"AÇÃO: {instr['name']}")
        elif op=="semaforo_action":
            sem = instr["semaforo"]; action = instr["action"]
            self.semaphores[sem]=action
            self.log.append(f"Semáforo {sem} -> {action}")
        elif op=="route_priority":
            cond = instr["condition"]
            if self.eval_expr(cond):
                key = instr["rota"]
                self.routes[key] = f"prioridade_{instr['priority']}"
                self.log.append(f"Rota {key} setada -> prioridade {instr['priority']}")
        elif op=="if":
            if self.eval_expr(instr["condition"]):
                for s in instr["body"]:
                    self.exec_stmt(s)
    def run(self):
        for instr in self.ir:
            self.exec_stmt(instr)
        return {"log":self.log,"vars":self.vars,"semaphores":self.semaphores,"routes":self.routes}

IR_DIR = "./ir_output"
IR_OUTPUT_PATH = os.path.join(IR_DIR, "ir_output.json")

def save_ir(ir: List[Dict[str,Any]], path: str = IR_OUTPUT_PATH) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ir, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("Falha ao salvar IR:", e)

def compile_pipeline(src: str, sensor_env: Optional[Dict[str,Any]]=None) -> Tuple[List[Token], Program, List[str], List[Dict[str,Any]]]:
    lexer = Lexer(src)
    toks = lexer.tokenize()
    parser = Parser(toks)
    ast = parser.parse()
    sem = SemanticAnalyzer(sensor_env)
    sem_errors = sem.analyze(ast)
    ir = IRGenerator().generate(ast)
    save_ir(ir)
    return toks, ast, sem_errors, ir


EX1 = """
# Exemplo 1 - semáforo action
sem_quadra abrir;
"""

EX2 = """
# Exemplo 2 - prioridade de rota condicional
alta rota_emergencial fluxo > 10;
"""

EX3 = """
# Exemplo 3 - if com set e comando
SET emergencia = true;
IF emergencia == true {
    sem_hospital intermitente;
    MOVE;
}
"""

def run_validation_examples():
    examples = [
        ("Exemplo Semáforo", EX1, {"fluxo": 5}),
        ("Exemplo Rota", EX2, {"fluxo": 12}),
        ("Exemplo If/Semáforo", EX3, {"fluxo": 0})
    ]
    for title, src, env in examples:
        print("\n\n======", title, "======")
        print(src.strip())
        toks, ast, sem_errors, ir = compile_pipeline(src, sensor_env=env)
        print("\nTokens:")
        print(toks)
        print("\nErros semânticos:")
        if sem_errors:
            for e in sem_errors: print("-", e)
        else:
            print("Nenhum erro.")
        print("\nIR gerado (JSON):")
        print(json.dumps(ir, indent=2, ensure_ascii=False))
        print(f"\nIR salvo automaticamente em: {IR_OUTPUT_PATH}")
        if sem_errors:
            print("Pulando simulação devido a erros semânticos.")
            continue
        sim = Simulator(ir, sensor_env=env)
        res = sim.run()
        print("\nSimulação (log):")
        for l in res["log"]:
            print("-", l)
        print("Estado final semáforos:", res["semaphores"])
        print("Estado final rotas:", res["routes"])
        print("Variáveis finais:", res["vars"])


def run_batch_tests(sources: List[Tuple[str,str,Dict[str,Any]]], max_tests: int = 50):
    executed = 0
    for title, src, env in sources:
        if executed >= max_tests: break
        print("\n\n*** Test:", title, "***")
        try:
            toks, ast, sem_errors, ir = compile_pipeline(src, sensor_env=env)
        except Exception as e:
            print("Erro durante compilação:", e)
            continue
        print("Sem erros semânticos." if not sem_errors else f"Erros: {sem_errors}")
        if not sem_errors:
            sim = Simulator(ir, sensor_env=env)
            res = sim.run()
            print("Resultado simulação:", res)
        executed += 1
    print(f"\nBatch finalizado. Executados: {executed}")


def main_menu():
    source = ""
    tokens=None; ast=None; semerrs=None; ir=None; sensors={}
    while True:
        print("\n=== MENU COMPILADORES ===")
        print("1 - Digitar código (termine com 'END')")
        print("2 - Carregar sensores (JSON ou caminho .json)")
        print("3 - Compilar e mostrar tokens/semântica/IR (IR salvo automaticamente)")
        print("4 - Simular (usa última IR e sensores carregados)")
        print("5 - Rodar batch de testes (padrão 3 exemplos, até 50)")
        print("0 - Sair")
        op = input("Opção: ").strip()
        if op=="1":
            print("Digite código (END para terminar):")
            lines=[]
            while True:
                try:
                    l=input()
                except EOFError:
                    l="END"
                if l.strip()=="END": break
                lines.append(l)
            source="\n".join(lines)
            print("Código carregado.")
        elif op=="2":
            raw = input("JSON sensores (ou caminho arquivo): ").strip()
            try:
                if raw.endswith(".json"):
                    with open(raw,"r",encoding="utf-8") as f:
                        sensors=json.load(f)
                else:
                    sensors=json.loads(raw)
                print("Sensores carregados:", sensors)
            except Exception as e:
                print("Erro ao carregar sensores:", e)
        elif op=="3":
            if not source:
                print("Nenhum código carregado.")
                continue
            try:
                tokens, ast, semerrs, ir = compile_pipeline(source, sensor_env=sensors)
            except Exception as e:
                print("Erro de parsing/compilação:", e)
                continue
            print("\nTOKENS:")
            for t in tokens: print(t)
            print("\nERROS SEMÂNTICOS:")
            if semerrs:
                for e in semerrs: print("-", e)
            else:
                print("Sem erros semânticos.")
            print("\nIR gerado:")
            print(json.dumps(ir, indent=2, ensure_ascii=False))
            print(f"IR salvo automaticamente em: {IR_OUTPUT_PATH}")
        elif op=="4":
            if ir is None:
                print("Compile o código primeiro (opção 3).")
                continue
            sim = Simulator(ir, sensor_env=sensors)
            res = sim.run()
            print("Log:")
            for l in res["log"]: print("-", l)
            print("Semáforos:", res["semaphores"])
            print("Rotas:", res["routes"])
            print("Vars:", res["vars"])
        elif op=="5":
            print("Rodando batch com os 3 exemplos internos (até 50 testes).")
            batch = [
                ("Ex1", EX1, {"fluxo":5}),
                ("Ex2", EX2, {"fluxo":12}),
                ("Ex3", EX3, {"fluxo":0})
            ]
            run_batch_tests(batch, max_tests=50)
        elif op=="0":
            print("Saindo..."); break
        else:
            print("Opção inválida.")

if __name__=="__main__":
    try:
        run_validation_examples()
        print("\n\nExemplos executados. Agora entrando no menu interativo.\n")
        main_menu()
    except KeyboardInterrupt:
        print("\nInterrompido. Saindo...")
        sys.exit(0)
