from KgeN.tir.ir.visitor import CollectVisitor


class AccessEntry:
    READ=0
    WRITE=1
    def __init__(self, type, tensor, nest_loop):
        self.type = type
        self.tensor = tensor
        self.nest_loop = nest_loop

class SyncAnalysisVisitor(CollectVisitor):
    def __init__(self):
        super().__init__()
        self.access_list = []
        self.nest_loop = []
        self.m = {
            AccessEntry.READ: {},
            AccessEntry.WRITE: {}
        }
    
    def analysis(self, func):
        func.accept(self)
        return func
    
    def visit_func_stmt(self, stmt):
        for st in stmt.body:
            st.accept(self)

        for entry in self.access_list:
            if entry.tensor in self.m[1 -  entry.type]:
                # write before
                loop, has_same_loop = find_sync_loop(entry.nest_loop, self.m[1 -  entry.type][entry.tensor].nest_loop)
                if has_same_loop:
                    loop.need_sync_before = True
                    loop.need_sync_after = True
                else:
                    loop.need_sync_before = True
            self.m[entry.type][entry.tensor] = entry
    
    def visit_for_stmt(self, stmt):
        self.nest_loop.append(stmt)
        for st in stmt.body:
            st.accept(self)
        self.nest_loop.pop()
    
    def visit_assign_stmt(self, stmt):
        tensor = stmt.dest.tensor
        for inp in tensor.inputs:
            if inp.scope == "shared":
                self.access_list.append(AccessEntry(AccessEntry.READ, inp, tuple(self.nest_loop)))
        if tensor.scope == "shared":
            self.access_list.append(AccessEntry(AccessEntry.WRITE, tensor, tuple(self.nest_loop)))

def find_sync_loop(loop_a, loop_b):
    has_same_loop = False
    for i, j in zip(loop_a, loop_b):
        if i is not j:
            return i, has_same_loop
        else:
            has_same_loop = True
    return None, has_same_loop

def sync_analysis_pass(func):
    visitor = SyncAnalysisVisitor()
    func = visitor.analysis(func)
    return func