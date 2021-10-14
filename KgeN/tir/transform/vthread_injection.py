from typing import Iterator
from KgeN.tir.ir.expr import IterVar
from KgeN.tir.ir.stmt import ForStmt
from KgeN.tir.ir.visitor import RewriteVisitor, CollectVisitor

class VthreadDetector(CollectVisitor):
    def __init__(self, target):
        self.target = target
        self.result = set()

    def detect(self, stmt):
        self.result = set()
        stmt.accept(self)
        return self.result

    def visit_iter_expr(self, expr):
        if expr in self.target:
            self.result.add(expr)
        return

class VThreadInjectionVisitor(RewriteVisitor):
    def __init__(self):
        super().__init__()
        self.vthread_set = set()
    
    def analysis(self, func):
        return func.accept(self)
    
    def visit_for_stmt(self, stmt):
        if stmt.iter_var.bind_to is not None and stmt.iter_var.bind_to.thread_tag == "vthread":
            self.vthread_set.add(stmt.iter_var)
        for i in range(len(stmt.body)):
            stmt.body[i] = stmt.body[i].accept(self)
        return stmt

    def visit_assign_stmt(self, stmt):
        detector = VthreadDetector(self.vthread_set)
        if len(detector.detect(stmt)) > 0:
            for iter in detector.result:
                new_iter = IterVar(iter.bind_to.name, iter.range.end, IterVar.DEFAULT)
                new_stmt = ForStmt(new_iter)
                new_stmt.body.append(stmt)
                stmt = new_stmt
        return stmt

def vthread_injection_pass(func):
    visitor = VThreadInjectionVisitor()
    func = visitor.analysis(func)
    return func