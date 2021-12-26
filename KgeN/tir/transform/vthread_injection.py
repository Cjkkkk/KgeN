from KgeN.tir.ir.expr import IterVar
from KgeN.tir.ir.stmt import ForStmt
from KgeN.tir.ir.visitor import RewriteVisitor, CollectVisitor
from KgeN.te.utils import axis_topo_sort_top_down


class VthreadDetector(CollectVisitor):
    def __init__(self):
        super().__init__()
        self.result = set()

    def detect(self, stmt):
        stmt.accept(self)
        return self.result

    def visit_iter_expr(self, expr):
        if expr.bind_to is not None and expr.bind_to.thread_tag == "vthread":
            self.result.add(expr)
        return

class VThreadInjectionVisitor(RewriteVisitor):
    def __init__(self):
        super().__init__()
        self.vthread_list = []
        self.rewrite_map = {}
    
    def analysis(self, func):
        return func.accept(self)

    def visit_for_stmt(self, stmt):
        if stmt.iter_var.bind_to is not None and stmt.iter_var.bind_to.thread_tag == "vthread":
            self.vthread_list.append(stmt.iter_var)
        for i in range(len(stmt.body)):
            stmt.body[i] = stmt.body[i].accept(self)
        return stmt
    
    def visit_tensor_slice_expr(self, expr):
        if expr.tensor in self.rewrite_map:
            shift = self.rewrite_map[expr.tensor]
            expr.index = (shift + expr.index[0],) + expr.index[1:]
        return expr

    def visit_assign_stmt(self, stmt):
        detector = VthreadDetector()
        detected_vthreads = detector.detect(stmt)
        detected_vthreads = [vthread for vthread in self.vthread_list if vthread in detected_vthreads]
        # see if stmt contains vthread iter, if true, add for loop to the stmt
        if len(detected_vthreads) > 0:
            tensor = stmt.dest.tensor
            ax = tensor.op.axis[0]
            shift = 0
            s = ax.range.end
            leaf_axis = axis_topo_sort_top_down(stmt.dest.tensor.op.axis)
            
            for iter in reversed(detected_vthreads):
                if iter in leaf_axis:
                    continue
                shift = iter * s + shift
                s *= iter.range.end

            stmt.dest.index = (shift + ax,) + stmt.dest.index[1:]
            tensor.shape = (s, ) + tensor.shape[1:]
            
            self.rewrite_map[tensor] = shift

            stmt.source = stmt.source.accept(self)
            
            for iter in detected_vthreads:
                new_iter = IterVar(iter.bind_to.name, iter.range.end, IterVar.DEFAULT)
                new_stmt = ForStmt(new_iter)
                new_stmt.body.append(stmt)
                stmt = new_stmt
        else:
            stmt.source = stmt.source.accept(self)
        return stmt

def vthread_injection_pass(func):
    visitor = VThreadInjectionVisitor()
    func = visitor.analysis(func)
    return func