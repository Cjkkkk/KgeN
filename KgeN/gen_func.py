from .tir import ForStmt, FuncStmt, AssignStmt, IterVar, Range, ReduceExpr, TensorSliceExpr, TensorExpr
from .utils import axis_topo_sort_top_down
from .schedule import compute_at

def add_reduce_init(tensor, fake_axis):
    attach_axis = fake_axis
    init_axis = []

    if isinstance(tensor.expr, ReduceExpr):
        # reduce expression
        all_reduce_axis = set(axis_topo_sort_top_down(tensor.reduce_axis))
        all_regular_axis = set(axis_topo_sort_top_down(tensor.root_axis))
        min_reduce_axis_idx = len(tensor.axis)
        for idx, axis in enumerate(tensor.axis):
            if axis in all_reduce_axis:
                min_reduce_axis_idx = idx
                break
        for idx, axis in enumerate(reversed(tensor.axis)):
            if axis in all_regular_axis:
                if idx > min_reduce_axis_idx:
                    attach_axis = axis
                    break
                else:
                    init_axis.append(axis)
        
        # TODO: fix this, use te.compute
        import copy
        init = copy.copy(tensor)
        copy_init_axis = copy.deepcopy(init_axis)
        init.axis = copy_init_axis
        init.expr = tensor.expr.init
        # avoid unintentional attach
        init.attached_computation = []
        compute_at(init, tensor, attach_axis)

def gen_stmt_for_tensor(tensor, stmt):
    def get_fake_axis():
        axis = IterVar("", 0, 0, Range.CLOSED_CLOSED)
        return axis
    fake_axis = get_fake_axis()
    # check if reduce init is needed
    add_reduce_init(tensor, fake_axis)
    
    # generate for stmt
    for axis in [fake_axis] + tensor.axis:
        new_stmt = ForStmt(axis)
        stmt.body.append(new_stmt)
        stmt = new_stmt

        for computation in axis.attached_computation:
            gen_stmt_for_tensor(computation, stmt)
    
    # generate assign stmt
    dest = TensorSliceExpr(tensor, tensor.root_axis)
    if isinstance(tensor.expr, ReduceExpr):
        source = tensor.expr.combinator(TensorSliceExpr(tensor, tensor.root_axis), tensor.expr.expr)
    else:
        source = tensor.expr
    new_stmt = AssignStmt(dest, source)
    stmt.body.append(new_stmt)
        
def gen_func_pass(tensors):
    func_stmt = FuncStmt()
    func_stmt.tensors = tensors
    func_stmt.input_tensors = [tensor for tensor in tensors if tensor.type == TensorExpr.PLACEHOLDER]
    func_stmt.output_tensors = [tensor for tensor in tensors if tensor.is_output()]
    for tensor in reversed(tensors):
        if tensor.type == TensorExpr.PLACEHOLDER or tensor.attached:
            continue
        gen_stmt_for_tensor(tensor, func_stmt)
    return func_stmt