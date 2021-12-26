from KgeN.te.operation import PlaceholderOp
from KgeN.tir.ir import ForStmt, FuncStmt, AssignStmt, IterVar, ReduceExpr, TensorSliceExpr, TensorExpr
from KgeN.te.utils import axis_topo_sort_top_down
from KgeN.te import Stage

def add_reduce_init(stage, fake_axis):
    tensor = stage.op.outputs[0]
    attach_axis = fake_axis
    init_axis = []

    if isinstance(stage.op.expr, ReduceExpr):
        # reduce expression
        all_reduce_axis = set(axis_topo_sort_top_down(tensor.op.reduce_axis))
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
        init_op = copy.copy(init.op)
        copy_init_axis = copy.deepcopy(init_axis)


        init.op = init_op
        init_stage = Stage(init_op)
        init_stage.leaf_axis = copy_init_axis
        init_stage.op.expr = tensor.op.expr.init
        
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
    tensor = stage.op.outputs[0]
    expr = stage.op.expr
    dest = TensorSliceExpr(tensor, stage.op.axis)
    if isinstance(expr, ReduceExpr):
        source = expr.combinator(dest, expr.expr)
    else:
        source = expr
    new_stmt = AssignStmt(dest, source)
    stmt.body.append(new_stmt)
        
def gen_func_pass(schdule, inputs, outputs):
    func_stmt = FuncStmt()
    func_stmt.schedule = schdule
    func_stmt.input_tensors = inputs
    func_stmt.output_tensors = outputs
    for stage in reversed(schdule.stages):
        if isinstance(stage.op, PlaceholderOp) or stage.attached:
            continue
        gen_stmt_for_stage(stage, func_stmt)
    return func_stmt