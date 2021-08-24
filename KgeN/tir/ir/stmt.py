class Stmt:
    def __init__(self):
        pass
    
    def __str__(self):
        from .ir_printer import IR_Printer
        ir_printer = IR_Printer()
        self.accept(ir_printer)
        return ir_printer.res

    def accept(self, visitor):
        raise NotImplementedError

class FuncStmt(Stmt):
    def __init__(self):
        super().__init__()
        self.body = []
        self.schedule = None
        self.input_tensors = []
        self.output_tensors = []

    def accept(self, visitor):
        return visitor.visit_func_stmt(self)

class IfStmt(Stmt):
    def __init__(self, condition, then_stmt, else_stmt=None):
        super().__init__()
        self.condition = condition
        self.then_stmt = then_stmt
        self.else_stmt = else_stmt

    def accept(self, visitor):
        return visitor.visit_if_stmt(self)
    
class ForStmt(Stmt):
    def __init__(self, iter_var):
        super().__init__()
        self.iter_var = iter_var
        self.body = []
        self.need_sync_before = False
        self.need_sync_after = False

    def accept(self, visitor):
        return visitor.visit_for_stmt(self)

class AssignStmt(Stmt):
    def __init__(self, dest, source):
        super().__init__()
        self.dest = dest
        self.source = source
    
    def accept(self, visitor):
        return visitor.visit_assign_stmt(self)