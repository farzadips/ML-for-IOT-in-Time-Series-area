??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
v
Dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameDense1/kernel
o
!Dense1/kernel/Read/ReadVariableOpReadVariableOpDense1/kernel*
_output_shapes

:*
dtype0
n
Dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameDense1/bias
g
Dense1/bias/Read/ReadVariableOpReadVariableOpDense1/bias*
_output_shapes
:*
dtype0
v
Dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameDense2/kernel
o
!Dense2/kernel/Read/ReadVariableOpReadVariableOpDense2/kernel*
_output_shapes

:*
dtype0
n
Dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameDense2/bias
g
Dense2/bias/Read/ReadVariableOpReadVariableOpDense2/bias*
_output_shapes
:*
dtype0
?
Output_layes/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameOutput_layes/kernel
{
'Output_layes/kernel/Read/ReadVariableOpReadVariableOpOutput_layes/kernel*
_output_shapes

:*
dtype0
z
Output_layes/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameOutput_layes/bias
s
%Output_layes/bias/Read/ReadVariableOpReadVariableOpOutput_layes/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
f
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	total_1
_
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
:*
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
	optimizer

signatures
#_self_saveable_object_factories
		variables

trainable_variables
regularization_losses
	keras_api
w
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
?

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
?

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
?

 kernel
!bias
#"_self_saveable_object_factories
#	variables
$trainable_variables
%regularization_losses
&	keras_api
w
#'_self_saveable_object_factories
(	variables
)trainable_variables
*regularization_losses
+	keras_api
 
 
 
*
0
1
2
3
 4
!5
*
0
1
2
3
 4
!5
 
?
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
		variables

trainable_variables
regularization_losses
 
 
 
 
?
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
YW
VARIABLE_VALUEDense1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEDense1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
?
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
YW
VARIABLE_VALUEDense2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEDense2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
 
?
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
_]
VARIABLE_VALUEOutput_layes/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEOutput_layes/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
 
?
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
#	variables
$trainable_variables
%regularization_losses
 
 
 
 
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
(	variables
)trainable_variables
*regularization_losses
 
#
0
1
2
3
4

J0
K1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ltotal
	Mcount
N	variables
O	keras_api
4
	Ptotal
	Qcount
R	variables
S	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

N	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1

R	variables
j
serving_default_xPlaceholder*"
_output_shapes
:*
dtype0*
shape:
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_xDense1/kernelDense1/biasDense2/kernelDense2/biasOutput_layes/kernelOutput_layes/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_59545
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!Dense1/kernel/Read/ReadVariableOpDense1/bias/Read/ReadVariableOp!Dense2/kernel/Read/ReadVariableOpDense2/bias/Read/ReadVariableOp'Output_layes/kernel/Read/ReadVariableOp%Output_layes/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_60078
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameDense1/kernelDense1/biasDense2/kernelDense2/biasOutput_layes/kernelOutput_layes/biastotalcounttotal_1count_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_60118??
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_59662

inputs
dense1_59608:
dense1_59610:
dense2_59625:
dense2_59627:$
output_layes_59641: 
output_layes_59643:
identity??Dense1/StatefulPartitionedCall?Dense2/StatefulPartitionedCall?$Output_layes/StatefulPartitionedCall?
Flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_59594?
Dense1/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0dense1_59608dense1_59610*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Dense1_layer_call_and_return_conditional_losses_59607?
Dense2/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0dense2_59625dense2_59627*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Dense2_layer_call_and_return_conditional_losses_59624?
$Output_layes/StatefulPartitionedCallStatefulPartitionedCall'Dense2/StatefulPartitionedCall:output:0output_layes_59641output_layes_59643*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Output_layes_layer_call_and_return_conditional_losses_59640?
reshape/PartitionedCallPartitionedCall-Output_layes/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_59659s
IdentityIdentity reshape/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^Dense1/StatefulPartitionedCall^Dense2/StatefulPartitionedCall%^Output_layes/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall2@
Dense2/StatefulPartitionedCallDense2/StatefulPartitionedCall2L
$Output_layes/StatefulPartitionedCall$Output_layes/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
A__inference_Dense1_layer_call_and_return_conditional_losses_59607

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

^
B__inference_reshape_layer_call_and_return_conditional_losses_59659

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
G__inference_Output_layes_layer_call_and_return_conditional_losses_59640

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_Dense1_layer_call_fn_59957

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Dense1_layer_call_and_return_conditional_losses_59607o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference__traced_save_60078
file_prefix,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop,
(savev2_dense2_kernel_read_readvariableop*
&savev2_dense2_bias_read_readvariableop2
.savev2_output_layes_kernel_read_readvariableop0
,savev2_output_layes_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableop.savev2_output_layes_kernel_read_readvariableop,savev2_output_layes_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*S
_input_shapesB
@: ::::::: : :: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: : 	

_output_shapes
::


_output_shapes
: :

_output_shapes
: 
?

?
A__inference_Dense2_layer_call_and_return_conditional_losses_59988

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_59867

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_59759s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
A__inference_Dense1_layer_call_and_return_conditional_losses_59968

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?,
?
__inference_<lambda>_59526
xB
0sequential_dense1_matmul_readvariableop_resource:?
1sequential_dense1_biasadd_readvariableop_resource:B
0sequential_dense2_matmul_readvariableop_resource:?
1sequential_dense2_biasadd_readvariableop_resource:H
6sequential_output_layes_matmul_readvariableop_resource:E
7sequential_output_layes_biasadd_readvariableop_resource:
identity??(sequential/Dense1/BiasAdd/ReadVariableOp?'sequential/Dense1/MatMul/ReadVariableOp?(sequential/Dense2/BiasAdd/ReadVariableOp?'sequential/Dense2/MatMul/ReadVariableOp?.sequential/Output_layes/BiasAdd/ReadVariableOp?-sequential/Output_layes/MatMul/ReadVariableOpi
sequential/Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   t
sequential/Flatten/ReshapeReshapex!sequential/Flatten/Const:output:0*
T0*
_output_shapes

:?
'sequential/Dense1/MatMul/ReadVariableOpReadVariableOp0sequential_dense1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential/Dense1/MatMulMatMul#sequential/Flatten/Reshape:output:0/sequential/Dense1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:?
(sequential/Dense1/BiasAdd/ReadVariableOpReadVariableOp1sequential_dense1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/Dense1/BiasAddBiasAdd"sequential/Dense1/MatMul:product:00sequential/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:k
sequential/Dense1/ReluRelu"sequential/Dense1/BiasAdd:output:0*
T0*
_output_shapes

:?
'sequential/Dense2/MatMul/ReadVariableOpReadVariableOp0sequential_dense2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential/Dense2/MatMulMatMul$sequential/Dense1/Relu:activations:0/sequential/Dense2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:?
(sequential/Dense2/BiasAdd/ReadVariableOpReadVariableOp1sequential_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/Dense2/BiasAddBiasAdd"sequential/Dense2/MatMul:product:00sequential/Dense2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:k
sequential/Dense2/ReluRelu"sequential/Dense2/BiasAdd:output:0*
T0*
_output_shapes

:?
-sequential/Output_layes/MatMul/ReadVariableOpReadVariableOp6sequential_output_layes_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential/Output_layes/MatMulMatMul$sequential/Dense2/Relu:activations:05sequential/Output_layes/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:?
.sequential/Output_layes/BiasAdd/ReadVariableOpReadVariableOp7sequential_output_layes_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/Output_layes/BiasAddBiasAdd(sequential/Output_layes/MatMul:product:06sequential/Output_layes/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:i
sequential/reshape/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      p
&sequential/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sequential/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sequential/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 sequential/reshape/strided_sliceStridedSlice!sequential/reshape/Shape:output:0/sequential/reshape/strided_slice/stack:output:01sequential/reshape/strided_slice/stack_1:output:01sequential/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d
"sequential/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
 sequential/reshape/Reshape/shapePack)sequential/reshape/strided_slice:output:0+sequential/reshape/Reshape/shape/1:output:0+sequential/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
sequential/reshape/ReshapeReshape(sequential/Output_layes/BiasAdd:output:0)sequential/reshape/Reshape/shape:output:0*
T0*"
_output_shapes
:m
IdentityIdentity#sequential/reshape/Reshape:output:0^NoOp*
T0*"
_output_shapes
:?
NoOpNoOp)^sequential/Dense1/BiasAdd/ReadVariableOp(^sequential/Dense1/MatMul/ReadVariableOp)^sequential/Dense2/BiasAdd/ReadVariableOp(^sequential/Dense2/MatMul/ReadVariableOp/^sequential/Output_layes/BiasAdd/ReadVariableOp.^sequential/Output_layes/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 2T
(sequential/Dense1/BiasAdd/ReadVariableOp(sequential/Dense1/BiasAdd/ReadVariableOp2R
'sequential/Dense1/MatMul/ReadVariableOp'sequential/Dense1/MatMul/ReadVariableOp2T
(sequential/Dense2/BiasAdd/ReadVariableOp(sequential/Dense2/BiasAdd/ReadVariableOp2R
'sequential/Dense2/MatMul/ReadVariableOp'sequential/Dense2/MatMul/ReadVariableOp2`
.sequential/Output_layes/BiasAdd/ReadVariableOp.sequential/Output_layes/BiasAdd/ReadVariableOp2^
-sequential/Output_layes/MatMul/ReadVariableOp-sequential/Output_layes/MatMul/ReadVariableOp:E A
"
_output_shapes
:

_user_specified_namex
?
?
,__inference_Output_layes_layer_call_fn_59997

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Output_layes_layer_call_and_return_conditional_losses_59640o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
 __inference__wrapped_model_59581
flatten_inputB
0sequential_dense1_matmul_readvariableop_resource:?
1sequential_dense1_biasadd_readvariableop_resource:B
0sequential_dense2_matmul_readvariableop_resource:?
1sequential_dense2_biasadd_readvariableop_resource:H
6sequential_output_layes_matmul_readvariableop_resource:E
7sequential_output_layes_biasadd_readvariableop_resource:
identity??(sequential/Dense1/BiasAdd/ReadVariableOp?'sequential/Dense1/MatMul/ReadVariableOp?(sequential/Dense2/BiasAdd/ReadVariableOp?'sequential/Dense2/MatMul/ReadVariableOp?.sequential/Output_layes/BiasAdd/ReadVariableOp?-sequential/Output_layes/MatMul/ReadVariableOpi
sequential/Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
sequential/Flatten/ReshapeReshapeflatten_input!sequential/Flatten/Const:output:0*
T0*'
_output_shapes
:??????????
'sequential/Dense1/MatMul/ReadVariableOpReadVariableOp0sequential_dense1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential/Dense1/MatMulMatMul#sequential/Flatten/Reshape:output:0/sequential/Dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(sequential/Dense1/BiasAdd/ReadVariableOpReadVariableOp1sequential_dense1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/Dense1/BiasAddBiasAdd"sequential/Dense1/MatMul:product:00sequential/Dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
sequential/Dense1/ReluRelu"sequential/Dense1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
'sequential/Dense2/MatMul/ReadVariableOpReadVariableOp0sequential_dense2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential/Dense2/MatMulMatMul$sequential/Dense1/Relu:activations:0/sequential/Dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
(sequential/Dense2/BiasAdd/ReadVariableOpReadVariableOp1sequential_dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/Dense2/BiasAddBiasAdd"sequential/Dense2/MatMul:product:00sequential/Dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????t
sequential/Dense2/ReluRelu"sequential/Dense2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
-sequential/Output_layes/MatMul/ReadVariableOpReadVariableOp6sequential_output_layes_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
sequential/Output_layes/MatMulMatMul$sequential/Dense2/Relu:activations:05sequential/Output_layes/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
.sequential/Output_layes/BiasAdd/ReadVariableOpReadVariableOp7sequential_output_layes_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential/Output_layes/BiasAddBiasAdd(sequential/Output_layes/MatMul:product:06sequential/Output_layes/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????p
sequential/reshape/ShapeShape(sequential/Output_layes/BiasAdd:output:0*
T0*
_output_shapes
:p
&sequential/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sequential/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sequential/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 sequential/reshape/strided_sliceStridedSlice!sequential/reshape/Shape:output:0/sequential/reshape/strided_slice/stack:output:01sequential/reshape/strided_slice/stack_1:output:01sequential/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :d
"sequential/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
 sequential/reshape/Reshape/shapePack)sequential/reshape/strided_slice:output:0+sequential/reshape/Reshape/shape/1:output:0+sequential/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
sequential/reshape/ReshapeReshape(sequential/Output_layes/BiasAdd:output:0)sequential/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????v
IdentityIdentity#sequential/reshape/Reshape:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp)^sequential/Dense1/BiasAdd/ReadVariableOp(^sequential/Dense1/MatMul/ReadVariableOp)^sequential/Dense2/BiasAdd/ReadVariableOp(^sequential/Dense2/MatMul/ReadVariableOp/^sequential/Output_layes/BiasAdd/ReadVariableOp.^sequential/Output_layes/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2T
(sequential/Dense1/BiasAdd/ReadVariableOp(sequential/Dense1/BiasAdd/ReadVariableOp2R
'sequential/Dense1/MatMul/ReadVariableOp'sequential/Dense1/MatMul/ReadVariableOp2T
(sequential/Dense2/BiasAdd/ReadVariableOp(sequential/Dense2/BiasAdd/ReadVariableOp2R
'sequential/Dense2/MatMul/ReadVariableOp'sequential/Dense2/MatMul/ReadVariableOp2`
.sequential/Output_layes/BiasAdd/ReadVariableOp.sequential/Output_layes/BiasAdd/ReadVariableOp2^
-sequential/Output_layes/MatMul/ReadVariableOp-sequential/Output_layes/MatMul/ReadVariableOp:Z V
+
_output_shapes
:?????????
'
_user_specified_nameFlatten_input
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_59759

inputs
dense1_59742:
dense1_59744:
dense2_59747:
dense2_59749:$
output_layes_59752: 
output_layes_59754:
identity??Dense1/StatefulPartitionedCall?Dense2/StatefulPartitionedCall?$Output_layes/StatefulPartitionedCall?
Flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_59594?
Dense1/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0dense1_59742dense1_59744*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Dense1_layer_call_and_return_conditional_losses_59607?
Dense2/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0dense2_59747dense2_59749*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Dense2_layer_call_and_return_conditional_losses_59624?
$Output_layes/StatefulPartitionedCallStatefulPartitionedCall'Dense2/StatefulPartitionedCall:output:0output_layes_59752output_layes_59754*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Output_layes_layer_call_and_return_conditional_losses_59640?
reshape/PartitionedCallPartitionedCall-Output_layes/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_59659s
IdentityIdentity reshape/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^Dense1/StatefulPartitionedCall^Dense2/StatefulPartitionedCall%^Output_layes/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall2@
Dense2/StatefulPartitionedCallDense2/StatefulPartitionedCall2L
$Output_layes/StatefulPartitionedCall$Output_layes/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_59833
flatten_input
dense1_59816:
dense1_59818:
dense2_59821:
dense2_59823:$
output_layes_59826: 
output_layes_59828:
identity??Dense1/StatefulPartitionedCall?Dense2/StatefulPartitionedCall?$Output_layes/StatefulPartitionedCall?
Flatten/PartitionedCallPartitionedCallflatten_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_59594?
Dense1/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0dense1_59816dense1_59818*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Dense1_layer_call_and_return_conditional_losses_59607?
Dense2/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0dense2_59821dense2_59823*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Dense2_layer_call_and_return_conditional_losses_59624?
$Output_layes/StatefulPartitionedCallStatefulPartitionedCall'Dense2/StatefulPartitionedCall:output:0output_layes_59826output_layes_59828*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Output_layes_layer_call_and_return_conditional_losses_59640?
reshape/PartitionedCallPartitionedCall-Output_layes/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_59659s
IdentityIdentity reshape/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^Dense1/StatefulPartitionedCall^Dense2/StatefulPartitionedCall%^Output_layes/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall2@
Dense2/StatefulPartitionedCallDense2/StatefulPartitionedCall2L
$Output_layes/StatefulPartitionedCall$Output_layes/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_nameFlatten_input
?
^
B__inference_Flatten_layer_call_and_return_conditional_losses_59594

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
E__inference_sequential_layer_call_and_return_conditional_losses_59937

inputs7
%dense1_matmul_readvariableop_resource:4
&dense1_biasadd_readvariableop_resource:7
%dense2_matmul_readvariableop_resource:4
&dense2_biasadd_readvariableop_resource:=
+output_layes_matmul_readvariableop_resource::
,output_layes_biasadd_readvariableop_resource:
identity??Dense1/BiasAdd/ReadVariableOp?Dense1/MatMul/ReadVariableOp?Dense2/BiasAdd/ReadVariableOp?Dense2/MatMul/ReadVariableOp?#Output_layes/BiasAdd/ReadVariableOp?"Output_layes/MatMul/ReadVariableOp^
Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   l
Flatten/ReshapeReshapeinputsFlatten/Const:output:0*
T0*'
_output_shapes
:??????????
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
Dense1/MatMulMatMulFlatten/Reshape:output:0$Dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
Dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
Dense2/MatMulMatMulDense1/Relu:activations:0$Dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Dense2/BiasAddBiasAddDense2/MatMul:product:0%Dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
Dense2/ReluReluDense2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
"Output_layes/MatMul/ReadVariableOpReadVariableOp+output_layes_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
Output_layes/MatMulMatMulDense2/Relu:activations:0*Output_layes/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#Output_layes/BiasAdd/ReadVariableOpReadVariableOp,output_layes_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Output_layes/BiasAddBiasAddOutput_layes/MatMul:product:0+Output_layes/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z
reshape/ShapeShapeOutput_layes/BiasAdd:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapeOutput_layes/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????k
IdentityIdentityreshape/Reshape:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^Dense1/BiasAdd/ReadVariableOp^Dense1/MatMul/ReadVariableOp^Dense2/BiasAdd/ReadVariableOp^Dense2/MatMul/ReadVariableOp$^Output_layes/BiasAdd/ReadVariableOp#^Output_layes/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2>
Dense1/BiasAdd/ReadVariableOpDense1/BiasAdd/ReadVariableOp2<
Dense1/MatMul/ReadVariableOpDense1/MatMul/ReadVariableOp2>
Dense2/BiasAdd/ReadVariableOpDense2/BiasAdd/ReadVariableOp2<
Dense2/MatMul/ReadVariableOpDense2/MatMul/ReadVariableOp2J
#Output_layes/BiasAdd/ReadVariableOp#Output_layes/BiasAdd/ReadVariableOp2H
"Output_layes/MatMul/ReadVariableOp"Output_layes/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
*__inference_sequential_layer_call_fn_59791
flatten_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_59759s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_nameFlatten_input
?
C
'__inference_Flatten_layer_call_fn_59942

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_59594`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
*__inference_sequential_layer_call_fn_59677
flatten_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_59662s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_nameFlatten_input
?
^
B__inference_Flatten_layer_call_and_return_conditional_losses_59948

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

^
B__inference_reshape_layer_call_and_return_conditional_losses_60025

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:?????????\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_59812
flatten_input
dense1_59795:
dense1_59797:
dense2_59800:
dense2_59802:$
output_layes_59805: 
output_layes_59807:
identity??Dense1/StatefulPartitionedCall?Dense2/StatefulPartitionedCall?$Output_layes/StatefulPartitionedCall?
Flatten/PartitionedCallPartitionedCallflatten_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_Flatten_layer_call_and_return_conditional_losses_59594?
Dense1/StatefulPartitionedCallStatefulPartitionedCall Flatten/PartitionedCall:output:0dense1_59795dense1_59797*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Dense1_layer_call_and_return_conditional_losses_59607?
Dense2/StatefulPartitionedCallStatefulPartitionedCall'Dense1/StatefulPartitionedCall:output:0dense2_59800dense2_59802*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Dense2_layer_call_and_return_conditional_losses_59624?
$Output_layes/StatefulPartitionedCallStatefulPartitionedCall'Dense2/StatefulPartitionedCall:output:0output_layes_59805output_layes_59807*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Output_layes_layer_call_and_return_conditional_losses_59640?
reshape/PartitionedCallPartitionedCall-Output_layes/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_59659s
IdentityIdentity reshape/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^Dense1/StatefulPartitionedCall^Dense2/StatefulPartitionedCall%^Output_layes/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2@
Dense1/StatefulPartitionedCallDense1/StatefulPartitionedCall2@
Dense2/StatefulPartitionedCallDense2/StatefulPartitionedCall2L
$Output_layes/StatefulPartitionedCall$Output_layes/StatefulPartitionedCall:Z V
+
_output_shapes
:?????????
'
_user_specified_nameFlatten_input
?
C
'__inference_reshape_layer_call_fn_60012

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_59659d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_59545
x
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *"
_output_shapes
:*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *#
fR
__inference_<lambda>_59526j
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*"
_output_shapes
:`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:E A
"
_output_shapes
:

_user_specified_namex
?*
?
!__inference__traced_restore_60118
file_prefix0
assignvariableop_dense1_kernel:,
assignvariableop_1_dense1_bias:2
 assignvariableop_2_dense2_kernel:,
assignvariableop_3_dense2_bias:8
&assignvariableop_4_output_layes_kernel:2
$assignvariableop_5_output_layes_bias:"
assignvariableop_6_total: "
assignvariableop_7_count: (
assignvariableop_8_total_1:$
assignvariableop_9_count_1: 
identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOpassignvariableop_dense1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp assignvariableop_2_dense2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp&assignvariableop_4_output_layes_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp$assignvariableop_5_output_layes_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOpassignvariableop_6_totalIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOpassignvariableop_7_countIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpassignvariableop_8_total_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOpassignvariableop_9_count_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
A__inference_Dense2_layer_call_and_return_conditional_losses_59624

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_sequential_layer_call_fn_59850

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_59662s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
E__inference_sequential_layer_call_and_return_conditional_losses_59902

inputs7
%dense1_matmul_readvariableop_resource:4
&dense1_biasadd_readvariableop_resource:7
%dense2_matmul_readvariableop_resource:4
&dense2_biasadd_readvariableop_resource:=
+output_layes_matmul_readvariableop_resource::
,output_layes_biasadd_readvariableop_resource:
identity??Dense1/BiasAdd/ReadVariableOp?Dense1/MatMul/ReadVariableOp?Dense2/BiasAdd/ReadVariableOp?Dense2/MatMul/ReadVariableOp?#Output_layes/BiasAdd/ReadVariableOp?"Output_layes/MatMul/ReadVariableOp^
Flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   l
Flatten/ReshapeReshapeinputsFlatten/Const:output:0*
T0*'
_output_shapes
:??????????
Dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
Dense1/MatMulMatMulFlatten/Reshape:output:0$Dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Dense1/BiasAdd/ReadVariableOpReadVariableOp&dense1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Dense1/BiasAddBiasAddDense1/MatMul:product:0%Dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
Dense1/ReluReluDense1/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
Dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
Dense2/MatMulMatMulDense1/Relu:activations:0$Dense2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
Dense2/BiasAdd/ReadVariableOpReadVariableOp&dense2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Dense2/BiasAddBiasAddDense2/MatMul:product:0%Dense2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????^
Dense2/ReluReluDense2/BiasAdd:output:0*
T0*'
_output_shapes
:??????????
"Output_layes/MatMul/ReadVariableOpReadVariableOp+output_layes_matmul_readvariableop_resource*
_output_shapes

:*
dtype0?
Output_layes/MatMulMatMulDense2/Relu:activations:0*Output_layes/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#Output_layes/BiasAdd/ReadVariableOpReadVariableOp,output_layes_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Output_layes/BiasAddBiasAddOutput_layes/MatMul:product:0+Output_layes/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????Z
reshape/ShapeShapeOutput_layes/BiasAdd:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:?
reshape/ReshapeReshapeOutput_layes/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:?????????k
IdentityIdentityreshape/Reshape:output:0^NoOp*
T0*+
_output_shapes
:??????????
NoOpNoOp^Dense1/BiasAdd/ReadVariableOp^Dense1/MatMul/ReadVariableOp^Dense2/BiasAdd/ReadVariableOp^Dense2/MatMul/ReadVariableOp$^Output_layes/BiasAdd/ReadVariableOp#^Output_layes/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2>
Dense1/BiasAdd/ReadVariableOpDense1/BiasAdd/ReadVariableOp2<
Dense1/MatMul/ReadVariableOpDense1/MatMul/ReadVariableOp2>
Dense2/BiasAdd/ReadVariableOpDense2/BiasAdd/ReadVariableOp2<
Dense2/MatMul/ReadVariableOpDense2/MatMul/ReadVariableOp2J
#Output_layes/BiasAdd/ReadVariableOp#Output_layes/BiasAdd/ReadVariableOp2H
"Output_layes/MatMul/ReadVariableOp"Output_layes/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_Dense2_layer_call_fn_59977

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_Dense2_layer_call_and_return_conditional_losses_59624o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
G__inference_Output_layes_layer_call_and_return_conditional_losses_60007

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
*
x%
serving_default_x:07
output_0+
StatefulPartitionedCall:0tensorflow/serving/predict:?c
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
	optimizer

signatures
#_self_saveable_object_factories
		variables

trainable_variables
regularization_losses
	keras_api
T__call__
*U&call_and_return_all_conditional_losses
V_default_save_signature"
_tf_keras_sequential
?
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
W__call__
*X&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
#_self_saveable_object_factories
	variables
trainable_variables
regularization_losses
	keras_api
[__call__
*\&call_and_return_all_conditional_losses"
_tf_keras_layer
?

 kernel
!bias
#"_self_saveable_object_factories
#	variables
$trainable_variables
%regularization_losses
&	keras_api
]__call__
*^&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#'_self_saveable_object_factories
(	variables
)trainable_variables
*regularization_losses
+	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer
,
aserving_default"
signature_map
 "
trackable_dict_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
		variables

trainable_variables
regularization_losses
T__call__
V_default_save_signature
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
:2Dense1/kernel
:2Dense1/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
:2Dense2/kernel
:2Dense2/bias
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
%:#2Output_layes/kernel
:2Output_layes/bias
 "
trackable_dict_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
#	variables
$trainable_variables
%regularization_losses
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
(	variables
)trainable_variables
*regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ltotal
	Mcount
N	variables
O	keras_api"
_tf_keras_metric
N
	Ptotal
	Qcount
R	variables
S	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
L0
M1"
trackable_list_wrapper
-
N	variables"
_generic_user_object
: (2total
:  (2count
.
P0
Q1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
?2?
*__inference_sequential_layer_call_fn_59677
*__inference_sequential_layer_call_fn_59850
*__inference_sequential_layer_call_fn_59867
*__inference_sequential_layer_call_fn_59791?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_59902
E__inference_sequential_layer_call_and_return_conditional_losses_59937
E__inference_sequential_layer_call_and_return_conditional_losses_59812
E__inference_sequential_layer_call_and_return_conditional_losses_59833?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
 __inference__wrapped_model_59581Flatten_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_Flatten_layer_call_fn_59942?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_Flatten_layer_call_and_return_conditional_losses_59948?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_Dense1_layer_call_fn_59957?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_Dense1_layer_call_and_return_conditional_losses_59968?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_Dense2_layer_call_fn_59977?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_Dense2_layer_call_and_return_conditional_losses_59988?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_Output_layes_layer_call_fn_59997?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_Output_layes_layer_call_and_return_conditional_losses_60007?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_reshape_layer_call_fn_60012?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_reshape_layer_call_and_return_conditional_losses_60025?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_59545x"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
A__inference_Dense1_layer_call_and_return_conditional_losses_59968\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
&__inference_Dense1_layer_call_fn_59957O/?,
%?"
 ?
inputs?????????
? "???????????
A__inference_Dense2_layer_call_and_return_conditional_losses_59988\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
&__inference_Dense2_layer_call_fn_59977O/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_Flatten_layer_call_and_return_conditional_losses_59948\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????
? z
'__inference_Flatten_layer_call_fn_59942O3?0
)?&
$?!
inputs?????????
? "???????????
G__inference_Output_layes_layer_call_and_return_conditional_losses_60007\ !/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? 
,__inference_Output_layes_layer_call_fn_59997O !/?,
%?"
 ?
inputs?????????
? "???????????
 __inference__wrapped_model_59581{ !:?7
0?-
+?(
Flatten_input?????????
? "5?2
0
reshape%?"
reshape??????????
B__inference_reshape_layer_call_and_return_conditional_losses_60025\/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? z
'__inference_reshape_layer_call_fn_60012O/?,
%?"
 ?
inputs?????????
? "???????????
E__inference_sequential_layer_call_and_return_conditional_losses_59812w !B??
8?5
+?(
Flatten_input?????????
p 

 
? ")?&
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_59833w !B??
8?5
+?(
Flatten_input?????????
p

 
? ")?&
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_59902p !;?8
1?.
$?!
inputs?????????
p 

 
? ")?&
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_59937p !;?8
1?.
$?!
inputs?????????
p

 
? ")?&
?
0?????????
? ?
*__inference_sequential_layer_call_fn_59677j !B??
8?5
+?(
Flatten_input?????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_59791j !B??
8?5
+?(
Flatten_input?????????
p

 
? "???????????
*__inference_sequential_layer_call_fn_59850c !;?8
1?.
$?!
inputs?????????
p 

 
? "???????????
*__inference_sequential_layer_call_fn_59867c !;?8
1?.
$?!
inputs?????????
p

 
? "???????????
#__inference_signature_wrapper_59545d !*?'
? 
 ?

x?
x".?+
)
output_0?
output_0