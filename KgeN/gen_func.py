from .tir import ForStmt, FuncStmt, AssignStmt, IterVar, ReduceExpr, TensorSliceExpr, TensorExpr
from .utils import axis_topo_sort_top_down
from .schedule import compute_at

def add_reduce_init(tensor, fake_axis):
    attach_axis = fake_axis
    init_axis = []

    if isinstance(tensor.expr, ReduceExpr):
        # reduce expression
        all_reduce_axis = set(axis_topo_sort_top_down(tensor.reduce_axis))
        first_reduce_idx = 0
        for idx, axis in enumerate(tensor.axis):
            if axis in all_reduce_axis:
                first_reduce_idx = idx
                break
            else:
                attach_axis = axis
        for axis in tensor.axis[first_reduce_idx + 1:]:
            if axis not in all_reduce_axis:
                init_axis.append(axis)
        # TODO: fix this, use te.compute
        import copy
        init = copy.copy(tensor)
        copy_init_axis = copy.deepcopy(init_axis)
        init.axis = copy_init_axis
        init.expr = tensor.expr.init
        # avoid unintentional attach
        for axis in copy_init_axis:
            axis.attached_computation = []
        compute_at(init, tensor, attach_axis)

def gen_stmt_for_tensor(tensor, stmt):
    def get_fake_axis():
        axis = IterVar("", 0, 1)
        return axis
    fake_axis = get_fake_axis()
    # add fake axis to express compute at root
    tensor.axis = [fake_axis] + tensor.axis
    # check if reduce init is needed
    add_reduce_init(tensor, fake_axis)
    
    # generate for stmt
    for axis in tensor.axis:
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
        
def gen_func_pass(inputs, outputs, tensors):
    func_stmt = FuncStmt()
    func_stmt.storage = tensors
    func_stmt.input_tensors = inputs
    func_stmt.output_tensors = outputs
    for tensor in reversed(tensors):
        if tensor.type == TensorExpr.PLACEHOLDER or tensor.attached:
            continue
        gen_stmt_for_tensor(tensor, func_stmt)
    return func_stmt