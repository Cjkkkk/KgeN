from KgeN.tir.ir import ForStmt, FuncStmt, AssignStmt, IterVar, ReduceExpr, TensorSliceExpr, TensorExpr
from KgeN.te.utils import axis_topo_sort_top_down
from KgeN.te import Stage

def add_reduce_init(stage, fake_axis):
    tensor = stage.tensor
    attach_axis = fake_axis
    init_axis = []

    if isinstance(tensor.expr, ReduceExpr):
        # reduce expression
        all_reduce_axis = set(axis_topo_sort_top_down(tensor.reduce_axis))
        first_reduce_idx = 0
        for idx, axis in enumerate(stage.leaf_axis):
            if axis in all_reduce_axis:
                first_reduce_idx = idx
                break
            else:
                attach_axis = axis
        for axis in stage.leaf_axis[first_reduce_idx + 1:]:
            if axis not in all_reduce_axis:
                init_axis.append(axis)
        # TODO: fix this, use te.compute
        import copy
        init = copy.copy(tensor)
        init_stage = Stage(init)
        copy_init_axis = copy.deepcopy(init_axis)
        init_stage.leaf_axis = copy_init_axis
        init.expr = tensor.expr.init
        # avoid unintentional attach
        for axis in copy_init_axis:
            axis.attached_computation = []
        init_stage.compute_at(stage, attach_axis)

def gen_stmt_for_stage(stage, stmt):
    def get_fake_axis():
        axis = IterVar("", 1, IterVar.DEFAULT)
        return axis
    fake_axis = get_fake_axis()
    # add fake axis to express compute at root
    stage.leaf_axis = [fake_axis] + stage.leaf_axis
    # check if reduce init is needed
    add_reduce_init(stage, fake_axis)
    
    # generate for stmt
    for axis in stage.leaf_axis:
        new_stmt = ForStmt(axis)
        stmt.body.append(new_stmt)
        stmt = new_stmt

        for computation in axis.attached_computation:
            gen_stmt_for_stage(computation, stmt)
    
    # generate assign stmt
    tensor = stage.tensor
    dest = TensorSliceExpr(tensor, tensor.index)
    if isinstance(tensor.expr, ReduceExpr):
        source = tensor.expr.combinator(TensorSliceExpr(tensor, tensor.index), tensor.expr.expr)
    else:
        source = tensor.expr
    new_stmt = AssignStmt(dest, source)
    stmt.body.append(new_stmt)
        
def gen_func_pass(schdule, inputs, outputs):
    func_stmt = FuncStmt()
    func_stmt.schedule = schdule
    func_stmt.input_tensors = inputs
    func_stmt.output_tensors = outputs
    for stage in reversed(schdule.stages):
        if stage.tensor.type == TensorExpr.PLACEHOLDER or stage.attached:
            continue
        gen_stmt_for_stage(stage, func_stmt)
    return func_stmt