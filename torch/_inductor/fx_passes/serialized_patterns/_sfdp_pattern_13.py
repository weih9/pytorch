# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

import torch
import torch._inductor

aten = torch.ops.aten
prims = torch.ops.prims

from torch._inductor.pattern_matcher import (
   Arg,
   CallFunction,
   CallFunctionVarArgs,
   CallMethod,
   CallMethodVarArgs,
   CallModule,
   CallModuleVarArgs,
   ExclusiveKeywordArg,
   Ignored,
   KeywordArg,
   ListOf,
   MultiOutputPattern,
   PatternExpr,
   RepeatedExpr,
   _TargetArgsExpr,
   _TargetExpr,
   _TargetExprVarArgs,
)
rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)
bmm_default = CallFunction(aten.bmm.default, KeywordArg('query'), permute_default, _users=2)
amax_default = CallFunction(aten.amax.default, bmm_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, bmm_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=2)
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, div_Tensor)
mul_Scalar = CallFunction(aten.mul.Scalar, mul_Tensor, Ignored(), _users=2)
bmm_default_1 = CallFunction(aten.bmm.default, mul_Scalar, KeywordArg('value'))
alias_default = CallFunction(aten.alias.default, div_Tensor)
alias_default_1 = CallFunction(aten.alias.default, alias_default)
alias_default_2 = CallFunction(aten.alias.default, alias_default_1)
alias_default_3 = CallFunction(aten.alias.default, alias_default_2, _users=2)
neg_default = CallFunction(aten.neg.default, alias_default_3)
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
bmm_default_2 = CallFunction(aten.bmm.default, KeywordArg('tangents_1'), permute_default_1)
convert_element_type_default = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
mul_Scalar_1 = CallFunction(aten.mul.Scalar, convert_element_type_default, Ignored())
mul_Tensor_1 = CallFunction(aten.mul.Tensor, bmm_default_2, mul_Scalar_1)
clone_default = CallFunction(aten.clone.default, mul_Tensor_1, memory_format=torch.contiguous_format)
mul_Tensor_2 = CallFunction(aten.mul.Tensor, clone_default, alias_default_3, _users=2)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_2, Ignored(), True)
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_2, _users=2)
permute_default_2 = CallFunction(aten.permute.default, permute_default, Ignored())
bmm_default_3 = CallFunction(aten.bmm.default, fma_default, permute_default_2)
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, fma_default)
permute_default_4 = CallFunction(aten.permute.default, bmm_default_4, Ignored())
permute_default_5 = CallFunction(aten.permute.default, mul_Scalar, Ignored())
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, KeywordArg('tangents_1'))
_sfdp_pattern_13_training = MultiOutputPattern([bmm_default_1,
  bmm_default_3,
  permute_default_4,
  bmm_default_5,
  None
])


permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
bmm_default = CallFunction(aten.bmm.default, KeywordArg('query'), permute_default, _users=2)
amax_default = CallFunction(aten.amax.default, bmm_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, bmm_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
clone_default = CallFunction(aten.clone.default, div_Tensor)
_sfdp_pattern_13_inference = CallFunction(aten.bmm.default, clone_default, KeywordArg('value'), _users=0)


rand_default = CallFunction(aten.rand.default, Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
gt_Scalar = CallFunction(aten.gt.Scalar, rand_default, KeywordArg('dropout_p'), _users=2)
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)
bmm_default = CallFunction(aten.bmm.default, KeywordArg('query'), permute_default)
convert_element_type_default = CallFunction(prims.convert_element_type.default, bmm_default, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)
mul_Tensor = CallFunction(aten.mul.Tensor, gt_Scalar, convert_element_type_default_1)
mul_Scalar = CallFunction(aten.mul.Scalar, mul_Tensor, Ignored(), _users=2)
bmm_default_1 = CallFunction(aten.bmm.default, mul_Scalar, KeywordArg('value'))
alias_default = CallFunction(aten.alias.default, convert_element_type_default_1)
alias_default_1 = CallFunction(aten.alias.default, alias_default)
alias_default_2 = CallFunction(aten.alias.default, alias_default_1)
alias_default_3 = CallFunction(aten.alias.default, alias_default_2)
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, alias_default_3, Ignored(), _users=2)
neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
bmm_default_2 = CallFunction(aten.bmm.default, KeywordArg('tangents_1'), permute_default_1)
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, gt_Scalar, Ignored())
mul_Scalar_1 = CallFunction(aten.mul.Scalar, convert_element_type_default_3, Ignored())
mul_Tensor_1 = CallFunction(aten.mul.Tensor, bmm_default_2, mul_Scalar_1)
clone_default = CallFunction(aten.clone.default, mul_Tensor_1, memory_format=torch.contiguous_format)
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, clone_default, Ignored())
mul_Tensor_2 = CallFunction(aten.mul.Tensor, convert_element_type_default_4, convert_element_type_default_2, _users=2)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor_2, Ignored(), True)
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor_2)
convert_element_type_default_5 = CallFunction(prims.convert_element_type.default, fma_default, Ignored(), _users=2)
permute_default_2 = CallFunction(aten.permute.default, permute_default, Ignored())
bmm_default_3 = CallFunction(aten.bmm.default, convert_element_type_default_5, permute_default_2)
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, convert_element_type_default_5)
permute_default_4 = CallFunction(aten.permute.default, bmm_default_4, Ignored())
permute_default_5 = CallFunction(aten.permute.default, mul_Scalar, Ignored())
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, KeywordArg('tangents_1'))
_sfdp_pattern_13_half_training = MultiOutputPattern([bmm_default_1,
  bmm_default_3,
  permute_default_4,
  bmm_default_5,
  None
])


permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
bmm_default = CallFunction(aten.bmm.default, KeywordArg('query'), permute_default)
convert_element_type_default = CallFunction(prims.convert_element_type.default, bmm_default, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored())
clone_default = CallFunction(aten.clone.default, convert_element_type_default_1)
_sfdp_pattern_13_half_inference = CallFunction(aten.bmm.default, clone_default, KeywordArg('value'), _users=0)
