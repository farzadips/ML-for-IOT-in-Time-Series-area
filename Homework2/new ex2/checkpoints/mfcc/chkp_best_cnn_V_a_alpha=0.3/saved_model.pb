??
??
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
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
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
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
?
Conv2D-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:&* 
shared_nameConv2D-1/kernel
{
#Conv2D-1/kernel/Read/ReadVariableOpReadVariableOpConv2D-1/kernel*&
_output_shapes
:&*
dtype0
z
Btch_Norm-1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*"
shared_nameBtch_Norm-1/gamma
s
%Btch_Norm-1/gamma/Read/ReadVariableOpReadVariableOpBtch_Norm-1/gamma*
_output_shapes
:&*
dtype0
x
Btch_Norm-1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*!
shared_nameBtch_Norm-1/beta
q
$Btch_Norm-1/beta/Read/ReadVariableOpReadVariableOpBtch_Norm-1/beta*
_output_shapes
:&*
dtype0
?
Btch_Norm-1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*(
shared_nameBtch_Norm-1/moving_mean

+Btch_Norm-1/moving_mean/Read/ReadVariableOpReadVariableOpBtch_Norm-1/moving_mean*
_output_shapes
:&*
dtype0
?
Btch_Norm-1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*,
shared_nameBtch_Norm-1/moving_variance
?
/Btch_Norm-1/moving_variance/Read/ReadVariableOpReadVariableOpBtch_Norm-1/moving_variance*
_output_shapes
:&*
dtype0
?
Conv2D-2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:&&* 
shared_nameConv2D-2/kernel
{
#Conv2D-2/kernel/Read/ReadVariableOpReadVariableOpConv2D-2/kernel*&
_output_shapes
:&&*
dtype0
z
Btch_Norm-2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*"
shared_nameBtch_Norm-2/gamma
s
%Btch_Norm-2/gamma/Read/ReadVariableOpReadVariableOpBtch_Norm-2/gamma*
_output_shapes
:&*
dtype0
x
Btch_Norm-2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*!
shared_nameBtch_Norm-2/beta
q
$Btch_Norm-2/beta/Read/ReadVariableOpReadVariableOpBtch_Norm-2/beta*
_output_shapes
:&*
dtype0
?
Btch_Norm-2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*(
shared_nameBtch_Norm-2/moving_mean

+Btch_Norm-2/moving_mean/Read/ReadVariableOpReadVariableOpBtch_Norm-2/moving_mean*
_output_shapes
:&*
dtype0
?
Btch_Norm-2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*,
shared_nameBtch_Norm-2/moving_variance
?
/Btch_Norm-2/moving_variance/Read/ReadVariableOpReadVariableOpBtch_Norm-2/moving_variance*
_output_shapes
:&*
dtype0
?
Conv2D-3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:&&* 
shared_nameConv2D-3/kernel
{
#Conv2D-3/kernel/Read/ReadVariableOpReadVariableOpConv2D-3/kernel*&
_output_shapes
:&&*
dtype0
z
Btch_Norm-3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*"
shared_nameBtch_Norm-3/gamma
s
%Btch_Norm-3/gamma/Read/ReadVariableOpReadVariableOpBtch_Norm-3/gamma*
_output_shapes
:&*
dtype0
x
Btch_Norm-3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*!
shared_nameBtch_Norm-3/beta
q
$Btch_Norm-3/beta/Read/ReadVariableOpReadVariableOpBtch_Norm-3/beta*
_output_shapes
:&*
dtype0
?
Btch_Norm-3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*(
shared_nameBtch_Norm-3/moving_mean

+Btch_Norm-3/moving_mean/Read/ReadVariableOpReadVariableOpBtch_Norm-3/moving_mean*
_output_shapes
:&*
dtype0
?
Btch_Norm-3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*,
shared_nameBtch_Norm-3/moving_variance
?
/Btch_Norm-3/moving_variance/Read/ReadVariableOpReadVariableOpBtch_Norm-3/moving_variance*
_output_shapes
:&*
dtype0
?
Output-Layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:&*$
shared_nameOutput-Layer/kernel
{
'Output-Layer/kernel/Read/ReadVariableOpReadVariableOpOutput-Layer/kernel*
_output_shapes

:&*
dtype0
z
Output-Layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameOutput-Layer/bias
s
%Output-Layer/bias/Read/ReadVariableOpReadVariableOpOutput-Layer/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
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
?
Adam/Conv2D-1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*'
shared_nameAdam/Conv2D-1/kernel/m
?
*Adam/Conv2D-1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv2D-1/kernel/m*&
_output_shapes
:&*
dtype0
?
Adam/Btch_Norm-1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*)
shared_nameAdam/Btch_Norm-1/gamma/m
?
,Adam/Btch_Norm-1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-1/gamma/m*
_output_shapes
:&*
dtype0
?
Adam/Btch_Norm-1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*(
shared_nameAdam/Btch_Norm-1/beta/m

+Adam/Btch_Norm-1/beta/m/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-1/beta/m*
_output_shapes
:&*
dtype0
?
Adam/Conv2D-2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&&*'
shared_nameAdam/Conv2D-2/kernel/m
?
*Adam/Conv2D-2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv2D-2/kernel/m*&
_output_shapes
:&&*
dtype0
?
Adam/Btch_Norm-2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*)
shared_nameAdam/Btch_Norm-2/gamma/m
?
,Adam/Btch_Norm-2/gamma/m/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-2/gamma/m*
_output_shapes
:&*
dtype0
?
Adam/Btch_Norm-2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*(
shared_nameAdam/Btch_Norm-2/beta/m

+Adam/Btch_Norm-2/beta/m/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-2/beta/m*
_output_shapes
:&*
dtype0
?
Adam/Conv2D-3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&&*'
shared_nameAdam/Conv2D-3/kernel/m
?
*Adam/Conv2D-3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv2D-3/kernel/m*&
_output_shapes
:&&*
dtype0
?
Adam/Btch_Norm-3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*)
shared_nameAdam/Btch_Norm-3/gamma/m
?
,Adam/Btch_Norm-3/gamma/m/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-3/gamma/m*
_output_shapes
:&*
dtype0
?
Adam/Btch_Norm-3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*(
shared_nameAdam/Btch_Norm-3/beta/m

+Adam/Btch_Norm-3/beta/m/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-3/beta/m*
_output_shapes
:&*
dtype0
?
Adam/Output-Layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:&*+
shared_nameAdam/Output-Layer/kernel/m
?
.Adam/Output-Layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Output-Layer/kernel/m*
_output_shapes

:&*
dtype0
?
Adam/Output-Layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Output-Layer/bias/m
?
,Adam/Output-Layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/Output-Layer/bias/m*
_output_shapes
:*
dtype0
?
Adam/Conv2D-1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*'
shared_nameAdam/Conv2D-1/kernel/v
?
*Adam/Conv2D-1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv2D-1/kernel/v*&
_output_shapes
:&*
dtype0
?
Adam/Btch_Norm-1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*)
shared_nameAdam/Btch_Norm-1/gamma/v
?
,Adam/Btch_Norm-1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-1/gamma/v*
_output_shapes
:&*
dtype0
?
Adam/Btch_Norm-1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*(
shared_nameAdam/Btch_Norm-1/beta/v

+Adam/Btch_Norm-1/beta/v/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-1/beta/v*
_output_shapes
:&*
dtype0
?
Adam/Conv2D-2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&&*'
shared_nameAdam/Conv2D-2/kernel/v
?
*Adam/Conv2D-2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv2D-2/kernel/v*&
_output_shapes
:&&*
dtype0
?
Adam/Btch_Norm-2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*)
shared_nameAdam/Btch_Norm-2/gamma/v
?
,Adam/Btch_Norm-2/gamma/v/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-2/gamma/v*
_output_shapes
:&*
dtype0
?
Adam/Btch_Norm-2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*(
shared_nameAdam/Btch_Norm-2/beta/v

+Adam/Btch_Norm-2/beta/v/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-2/beta/v*
_output_shapes
:&*
dtype0
?
Adam/Conv2D-3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&&*'
shared_nameAdam/Conv2D-3/kernel/v
?
*Adam/Conv2D-3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv2D-3/kernel/v*&
_output_shapes
:&&*
dtype0
?
Adam/Btch_Norm-3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*)
shared_nameAdam/Btch_Norm-3/gamma/v
?
,Adam/Btch_Norm-3/gamma/v/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-3/gamma/v*
_output_shapes
:&*
dtype0
?
Adam/Btch_Norm-3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*(
shared_nameAdam/Btch_Norm-3/beta/v

+Adam/Btch_Norm-3/beta/v/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-3/beta/v*
_output_shapes
:&*
dtype0
?
Adam/Output-Layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:&*+
shared_nameAdam/Output-Layer/kernel/v
?
.Adam/Output-Layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Output-Layer/kernel/v*
_output_shapes

:&*
dtype0
?
Adam/Output-Layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Output-Layer/bias/v
?
,Adam/Output-Layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/Output-Layer/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?P
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?P
value?PB?P B?P
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer-9
layer_with_weights-6
layer-10
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
^

kernel
	variables
trainable_variables
regularization_losses
	keras_api
?
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
R
 	variables
!trainable_variables
"regularization_losses
#	keras_api
^

$kernel
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?
)axis
	*gamma
+beta
,moving_mean
-moving_variance
.	variables
/trainable_variables
0regularization_losses
1	keras_api
R
2	variables
3trainable_variables
4regularization_losses
5	keras_api
^

6kernel
7	variables
8trainable_variables
9regularization_losses
:	keras_api
?
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
R
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
R
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
h

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
?
Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratem?m?m?$m?*m?+m?6m?<m?=m?Lm?Mm?v?v?v?$v?*v?+v?6v?<v?=v?Lv?Mv?
~
0
1
2
3
4
$5
*6
+7
,8
-9
610
<11
=12
>13
?14
L15
M16
N
0
1
2
$3
*4
+5
66
<7
=8
L9
M10
 
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
 
[Y
VARIABLE_VALUEConv2D-1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
 
\Z
VARIABLE_VALUEBtch_Norm-1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEBtch_Norm-1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEBtch_Norm-1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEBtch_Norm-1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
 
?
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
 	variables
!trainable_variables
"regularization_losses
[Y
VARIABLE_VALUEConv2D-2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE

$0

$0
 
?
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
%	variables
&trainable_variables
'regularization_losses
 
\Z
VARIABLE_VALUEBtch_Norm-2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEBtch_Norm-2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEBtch_Norm-2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEBtch_Norm-2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

*0
+1
,2
-3

*0
+1
 
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
.	variables
/trainable_variables
0regularization_losses
 
 
 
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
2	variables
3trainable_variables
4regularization_losses
[Y
VARIABLE_VALUEConv2D-3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE

60

60
 
?
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
7	variables
8trainable_variables
9regularization_losses
 
\Z
VARIABLE_VALUEBtch_Norm-3/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEBtch_Norm-3/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEBtch_Norm-3/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEBtch_Norm-3/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

<0
=1
>2
?3

<0
=1
 
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
 
 
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
_]
VARIABLE_VALUEOutput-Layer/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEOutput-Layer/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

L0
M1
 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
*
0
1
,2
-3
>4
?5
N
0
1
2
3
4
5
6
7
	8

9
10

?0
?1
 
 
 
 
 
 
 

0
1
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

,0
-1
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

>0
?1
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
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
~|
VARIABLE_VALUEAdam/Conv2D-1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/Btch_Norm-1/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Btch_Norm-1/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv2D-2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/Btch_Norm-2/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Btch_Norm-2/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv2D-3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/Btch_Norm-3/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Btch_Norm-3/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Output-Layer/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Output-Layer/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv2D-1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/Btch_Norm-1/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Btch_Norm-1/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv2D-2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/Btch_Norm-2/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Btch_Norm-2/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Conv2D-3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/Btch_Norm-3/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/Btch_Norm-3/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/Output-Layer/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/Output-Layer/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_Conv2D-1_inputPlaceholder*/
_output_shapes
:?????????'
*
dtype0*$
shape:?????????'

?
StatefulPartitionedCallStatefulPartitionedCallserving_default_Conv2D-1_inputConv2D-1/kernelBtch_Norm-1/gammaBtch_Norm-1/betaBtch_Norm-1/moving_meanBtch_Norm-1/moving_varianceConv2D-2/kernelBtch_Norm-2/gammaBtch_Norm-2/betaBtch_Norm-2/moving_meanBtch_Norm-2/moving_varianceConv2D-3/kernelBtch_Norm-3/gammaBtch_Norm-3/betaBtch_Norm-3/moving_meanBtch_Norm-3/moving_varianceOutput-Layer/kernelOutput-Layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_128216
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#Conv2D-1/kernel/Read/ReadVariableOp%Btch_Norm-1/gamma/Read/ReadVariableOp$Btch_Norm-1/beta/Read/ReadVariableOp+Btch_Norm-1/moving_mean/Read/ReadVariableOp/Btch_Norm-1/moving_variance/Read/ReadVariableOp#Conv2D-2/kernel/Read/ReadVariableOp%Btch_Norm-2/gamma/Read/ReadVariableOp$Btch_Norm-2/beta/Read/ReadVariableOp+Btch_Norm-2/moving_mean/Read/ReadVariableOp/Btch_Norm-2/moving_variance/Read/ReadVariableOp#Conv2D-3/kernel/Read/ReadVariableOp%Btch_Norm-3/gamma/Read/ReadVariableOp$Btch_Norm-3/beta/Read/ReadVariableOp+Btch_Norm-3/moving_mean/Read/ReadVariableOp/Btch_Norm-3/moving_variance/Read/ReadVariableOp'Output-Layer/kernel/Read/ReadVariableOp%Output-Layer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/Conv2D-1/kernel/m/Read/ReadVariableOp,Adam/Btch_Norm-1/gamma/m/Read/ReadVariableOp+Adam/Btch_Norm-1/beta/m/Read/ReadVariableOp*Adam/Conv2D-2/kernel/m/Read/ReadVariableOp,Adam/Btch_Norm-2/gamma/m/Read/ReadVariableOp+Adam/Btch_Norm-2/beta/m/Read/ReadVariableOp*Adam/Conv2D-3/kernel/m/Read/ReadVariableOp,Adam/Btch_Norm-3/gamma/m/Read/ReadVariableOp+Adam/Btch_Norm-3/beta/m/Read/ReadVariableOp.Adam/Output-Layer/kernel/m/Read/ReadVariableOp,Adam/Output-Layer/bias/m/Read/ReadVariableOp*Adam/Conv2D-1/kernel/v/Read/ReadVariableOp,Adam/Btch_Norm-1/gamma/v/Read/ReadVariableOp+Adam/Btch_Norm-1/beta/v/Read/ReadVariableOp*Adam/Conv2D-2/kernel/v/Read/ReadVariableOp,Adam/Btch_Norm-2/gamma/v/Read/ReadVariableOp+Adam/Btch_Norm-2/beta/v/Read/ReadVariableOp*Adam/Conv2D-3/kernel/v/Read/ReadVariableOp,Adam/Btch_Norm-3/gamma/v/Read/ReadVariableOp+Adam/Btch_Norm-3/beta/v/Read/ReadVariableOp.Adam/Output-Layer/kernel/v/Read/ReadVariableOp,Adam/Output-Layer/bias/v/Read/ReadVariableOpConst*=
Tin6
422	*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_129078
?

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameConv2D-1/kernelBtch_Norm-1/gammaBtch_Norm-1/betaBtch_Norm-1/moving_meanBtch_Norm-1/moving_varianceConv2D-2/kernelBtch_Norm-2/gammaBtch_Norm-2/betaBtch_Norm-2/moving_meanBtch_Norm-2/moving_varianceConv2D-3/kernelBtch_Norm-3/gammaBtch_Norm-3/betaBtch_Norm-3/moving_meanBtch_Norm-3/moving_varianceOutput-Layer/kernelOutput-Layer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/Conv2D-1/kernel/mAdam/Btch_Norm-1/gamma/mAdam/Btch_Norm-1/beta/mAdam/Conv2D-2/kernel/mAdam/Btch_Norm-2/gamma/mAdam/Btch_Norm-2/beta/mAdam/Conv2D-3/kernel/mAdam/Btch_Norm-3/gamma/mAdam/Btch_Norm-3/beta/mAdam/Output-Layer/kernel/mAdam/Output-Layer/bias/mAdam/Conv2D-1/kernel/vAdam/Btch_Norm-1/gamma/vAdam/Btch_Norm-1/beta/vAdam/Conv2D-2/kernel/vAdam/Btch_Norm-2/gamma/vAdam/Btch_Norm-2/beta/vAdam/Conv2D-3/kernel/vAdam/Btch_Norm-3/gamma/vAdam/Btch_Norm-3/beta/vAdam/Output-Layer/kernel/vAdam/Output-Layer/bias/v*<
Tin5
321*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_129232??
?3
?
I__inference_sequential_25_layer_call_and_return_conditional_losses_128120
conv2d_1_input)
conv2d_1_128074:& 
btch_norm_1_128077:& 
btch_norm_1_128079:& 
btch_norm_1_128081:& 
btch_norm_1_128083:&)
conv2d_2_128087:&& 
btch_norm_2_128090:& 
btch_norm_2_128092:& 
btch_norm_2_128094:& 
btch_norm_2_128096:&)
conv2d_3_128100:&& 
btch_norm_3_128103:& 
btch_norm_3_128105:& 
btch_norm_3_128107:& 
btch_norm_3_128109:&%
output_layer_128114:&!
output_layer_128116:
identity??#Btch_Norm-1/StatefulPartitionedCall?#Btch_Norm-2/StatefulPartitionedCall?#Btch_Norm-3/StatefulPartitionedCall? Conv2D-1/StatefulPartitionedCall? Conv2D-2/StatefulPartitionedCall? Conv2D-3/StatefulPartitionedCall?$Output-Layer/StatefulPartitionedCall?
 Conv2D-1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_128074*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-1_layer_call_and_return_conditional_losses_127527?
#Btch_Norm-1/StatefulPartitionedCallStatefulPartitionedCall)Conv2D-1/StatefulPartitionedCall:output:0btch_norm_1_128077btch_norm_1_128079btch_norm_1_128081btch_norm_1_128083*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_127548?
re_lu_48/PartitionedCallPartitionedCall,Btch_Norm-1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_48_layer_call_and_return_conditional_losses_127563?
 Conv2D-2/StatefulPartitionedCallStatefulPartitionedCall!re_lu_48/PartitionedCall:output:0conv2d_2_128087*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-2_layer_call_and_return_conditional_losses_127572?
#Btch_Norm-2/StatefulPartitionedCallStatefulPartitionedCall)Conv2D-2/StatefulPartitionedCall:output:0btch_norm_2_128090btch_norm_2_128092btch_norm_2_128094btch_norm_2_128096*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_127593?
re_lu_49/PartitionedCallPartitionedCall,Btch_Norm-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_49_layer_call_and_return_conditional_losses_127608?
 Conv2D-3/StatefulPartitionedCallStatefulPartitionedCall!re_lu_49/PartitionedCall:output:0conv2d_3_128100*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-3_layer_call_and_return_conditional_losses_127617?
#Btch_Norm-3/StatefulPartitionedCallStatefulPartitionedCall)Conv2D-3/StatefulPartitionedCall:output:0btch_norm_3_128103btch_norm_3_128105btch_norm_3_128107btch_norm_3_128109*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_127638?
re_lu_50/PartitionedCallPartitionedCall,Btch_Norm-3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_50_layer_call_and_return_conditional_losses_127653?
*GlobalAveragePooling-Layer/PartitionedCallPartitionedCall!re_lu_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_127660?
$Output-Layer/StatefulPartitionedCallStatefulPartitionedCall3GlobalAveragePooling-Layer/PartitionedCall:output:0output_layer_128114output_layer_128116*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_Output-Layer_layer_call_and_return_conditional_losses_127672|
IdentityIdentity-Output-Layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp$^Btch_Norm-1/StatefulPartitionedCall$^Btch_Norm-2/StatefulPartitionedCall$^Btch_Norm-3/StatefulPartitionedCall!^Conv2D-1/StatefulPartitionedCall!^Conv2D-2/StatefulPartitionedCall!^Conv2D-3/StatefulPartitionedCall%^Output-Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????'
: : : : : : : : : : : : : : : : : 2J
#Btch_Norm-1/StatefulPartitionedCall#Btch_Norm-1/StatefulPartitionedCall2J
#Btch_Norm-2/StatefulPartitionedCall#Btch_Norm-2/StatefulPartitionedCall2J
#Btch_Norm-3/StatefulPartitionedCall#Btch_Norm-3/StatefulPartitionedCall2D
 Conv2D-1/StatefulPartitionedCall Conv2D-1/StatefulPartitionedCall2D
 Conv2D-2/StatefulPartitionedCall Conv2D-2/StatefulPartitionedCall2D
 Conv2D-3/StatefulPartitionedCall Conv2D-3/StatefulPartitionedCall2L
$Output-Layer/StatefulPartitionedCall$Output-Layer/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????'

(
_user_specified_nameConv2D-1_input
?
?
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_127638

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????&?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_127330

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
?
.__inference_sequential_25_layer_call_fn_127716
conv2d_1_input!
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
	unknown_3:&#
	unknown_4:&&
	unknown_5:&
	unknown_6:&
	unknown_7:&
	unknown_8:&#
	unknown_9:&&

unknown_10:&

unknown_11:&

unknown_12:&

unknown_13:&

unknown_14:&

unknown_15:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_25_layer_call_and_return_conditional_losses_127679o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????'
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????'

(
_user_specified_nameConv2D-1_input
?
?
D__inference_Conv2D-3_layer_call_and_return_conditional_losses_128736

inputs8
conv2d_readvariableop_resource:&&
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:&&*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????&^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????&: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
D__inference_Conv2D-1_layer_call_and_return_conditional_losses_128440

inputs8
conv2d_readvariableop_resource:&
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:&*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????&^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????'
: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????'

 
_user_specified_nameinputs
?
?
,__inference_Btch_Norm-2_layer_call_fn_128627

inputs
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_127593w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
.__inference_sequential_25_layer_call_fn_128255

inputs!
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
	unknown_3:&#
	unknown_4:&&
	unknown_5:&
	unknown_6:&
	unknown_7:&
	unknown_8:&#
	unknown_9:&&

unknown_10:&

unknown_11:&

unknown_12:&

unknown_13:&

unknown_14:&

unknown_15:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_25_layer_call_and_return_conditional_losses_127679o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????'
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????'

 
_user_specified_nameinputs
?
?
D__inference_Conv2D-1_layer_call_and_return_conditional_losses_127527

inputs8
conv2d_readvariableop_resource:&
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:&*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????&^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????'
: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????'

 
_user_specified_nameinputs
?Z
?
!__inference__wrapped_model_127308
conv2d_1_inputO
5sequential_25_conv2d_1_conv2d_readvariableop_resource:&?
1sequential_25_btch_norm_1_readvariableop_resource:&A
3sequential_25_btch_norm_1_readvariableop_1_resource:&P
Bsequential_25_btch_norm_1_fusedbatchnormv3_readvariableop_resource:&R
Dsequential_25_btch_norm_1_fusedbatchnormv3_readvariableop_1_resource:&O
5sequential_25_conv2d_2_conv2d_readvariableop_resource:&&?
1sequential_25_btch_norm_2_readvariableop_resource:&A
3sequential_25_btch_norm_2_readvariableop_1_resource:&P
Bsequential_25_btch_norm_2_fusedbatchnormv3_readvariableop_resource:&R
Dsequential_25_btch_norm_2_fusedbatchnormv3_readvariableop_1_resource:&O
5sequential_25_conv2d_3_conv2d_readvariableop_resource:&&?
1sequential_25_btch_norm_3_readvariableop_resource:&A
3sequential_25_btch_norm_3_readvariableop_1_resource:&P
Bsequential_25_btch_norm_3_fusedbatchnormv3_readvariableop_resource:&R
Dsequential_25_btch_norm_3_fusedbatchnormv3_readvariableop_1_resource:&K
9sequential_25_output_layer_matmul_readvariableop_resource:&H
:sequential_25_output_layer_biasadd_readvariableop_resource:
identity??9sequential_25/Btch_Norm-1/FusedBatchNormV3/ReadVariableOp?;sequential_25/Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1?(sequential_25/Btch_Norm-1/ReadVariableOp?*sequential_25/Btch_Norm-1/ReadVariableOp_1?9sequential_25/Btch_Norm-2/FusedBatchNormV3/ReadVariableOp?;sequential_25/Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1?(sequential_25/Btch_Norm-2/ReadVariableOp?*sequential_25/Btch_Norm-2/ReadVariableOp_1?9sequential_25/Btch_Norm-3/FusedBatchNormV3/ReadVariableOp?;sequential_25/Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1?(sequential_25/Btch_Norm-3/ReadVariableOp?*sequential_25/Btch_Norm-3/ReadVariableOp_1?,sequential_25/Conv2D-1/Conv2D/ReadVariableOp?,sequential_25/Conv2D-2/Conv2D/ReadVariableOp?,sequential_25/Conv2D-3/Conv2D/ReadVariableOp?1sequential_25/Output-Layer/BiasAdd/ReadVariableOp?0sequential_25/Output-Layer/MatMul/ReadVariableOp?
,sequential_25/Conv2D-1/Conv2D/ReadVariableOpReadVariableOp5sequential_25_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:&*
dtype0?
sequential_25/Conv2D-1/Conv2DConv2Dconv2d_1_input4sequential_25/Conv2D-1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
?
(sequential_25/Btch_Norm-1/ReadVariableOpReadVariableOp1sequential_25_btch_norm_1_readvariableop_resource*
_output_shapes
:&*
dtype0?
*sequential_25/Btch_Norm-1/ReadVariableOp_1ReadVariableOp3sequential_25_btch_norm_1_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
9sequential_25/Btch_Norm-1/FusedBatchNormV3/ReadVariableOpReadVariableOpBsequential_25_btch_norm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
;sequential_25/Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDsequential_25_btch_norm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
*sequential_25/Btch_Norm-1/FusedBatchNormV3FusedBatchNormV3&sequential_25/Conv2D-1/Conv2D:output:00sequential_25/Btch_Norm-1/ReadVariableOp:value:02sequential_25/Btch_Norm-1/ReadVariableOp_1:value:0Asequential_25/Btch_Norm-1/FusedBatchNormV3/ReadVariableOp:value:0Csequential_25/Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
is_training( ?
sequential_25/re_lu_48/ReluRelu.sequential_25/Btch_Norm-1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&?
,sequential_25/Conv2D-2/Conv2D/ReadVariableOpReadVariableOp5sequential_25_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:&&*
dtype0?
sequential_25/Conv2D-2/Conv2DConv2D)sequential_25/re_lu_48/Relu:activations:04sequential_25/Conv2D-2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
?
(sequential_25/Btch_Norm-2/ReadVariableOpReadVariableOp1sequential_25_btch_norm_2_readvariableop_resource*
_output_shapes
:&*
dtype0?
*sequential_25/Btch_Norm-2/ReadVariableOp_1ReadVariableOp3sequential_25_btch_norm_2_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
9sequential_25/Btch_Norm-2/FusedBatchNormV3/ReadVariableOpReadVariableOpBsequential_25_btch_norm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
;sequential_25/Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDsequential_25_btch_norm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
*sequential_25/Btch_Norm-2/FusedBatchNormV3FusedBatchNormV3&sequential_25/Conv2D-2/Conv2D:output:00sequential_25/Btch_Norm-2/ReadVariableOp:value:02sequential_25/Btch_Norm-2/ReadVariableOp_1:value:0Asequential_25/Btch_Norm-2/FusedBatchNormV3/ReadVariableOp:value:0Csequential_25/Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
is_training( ?
sequential_25/re_lu_49/ReluRelu.sequential_25/Btch_Norm-2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&?
,sequential_25/Conv2D-3/Conv2D/ReadVariableOpReadVariableOp5sequential_25_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:&&*
dtype0?
sequential_25/Conv2D-3/Conv2DConv2D)sequential_25/re_lu_49/Relu:activations:04sequential_25/Conv2D-3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
?
(sequential_25/Btch_Norm-3/ReadVariableOpReadVariableOp1sequential_25_btch_norm_3_readvariableop_resource*
_output_shapes
:&*
dtype0?
*sequential_25/Btch_Norm-3/ReadVariableOp_1ReadVariableOp3sequential_25_btch_norm_3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
9sequential_25/Btch_Norm-3/FusedBatchNormV3/ReadVariableOpReadVariableOpBsequential_25_btch_norm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
;sequential_25/Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpDsequential_25_btch_norm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
*sequential_25/Btch_Norm-3/FusedBatchNormV3FusedBatchNormV3&sequential_25/Conv2D-3/Conv2D:output:00sequential_25/Btch_Norm-3/ReadVariableOp:value:02sequential_25/Btch_Norm-3/ReadVariableOp_1:value:0Asequential_25/Btch_Norm-3/FusedBatchNormV3/ReadVariableOp:value:0Csequential_25/Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
is_training( ?
sequential_25/re_lu_50/ReluRelu.sequential_25/Btch_Norm-3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&?
?sequential_25/GlobalAveragePooling-Layer/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
-sequential_25/GlobalAveragePooling-Layer/MeanMean)sequential_25/re_lu_50/Relu:activations:0Hsequential_25/GlobalAveragePooling-Layer/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????&?
0sequential_25/Output-Layer/MatMul/ReadVariableOpReadVariableOp9sequential_25_output_layer_matmul_readvariableop_resource*
_output_shapes

:&*
dtype0?
!sequential_25/Output-Layer/MatMulMatMul6sequential_25/GlobalAveragePooling-Layer/Mean:output:08sequential_25/Output-Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
1sequential_25/Output-Layer/BiasAdd/ReadVariableOpReadVariableOp:sequential_25_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
"sequential_25/Output-Layer/BiasAddBiasAdd+sequential_25/Output-Layer/MatMul:product:09sequential_25/Output-Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????z
IdentityIdentity+sequential_25/Output-Layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp:^sequential_25/Btch_Norm-1/FusedBatchNormV3/ReadVariableOp<^sequential_25/Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1)^sequential_25/Btch_Norm-1/ReadVariableOp+^sequential_25/Btch_Norm-1/ReadVariableOp_1:^sequential_25/Btch_Norm-2/FusedBatchNormV3/ReadVariableOp<^sequential_25/Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1)^sequential_25/Btch_Norm-2/ReadVariableOp+^sequential_25/Btch_Norm-2/ReadVariableOp_1:^sequential_25/Btch_Norm-3/FusedBatchNormV3/ReadVariableOp<^sequential_25/Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1)^sequential_25/Btch_Norm-3/ReadVariableOp+^sequential_25/Btch_Norm-3/ReadVariableOp_1-^sequential_25/Conv2D-1/Conv2D/ReadVariableOp-^sequential_25/Conv2D-2/Conv2D/ReadVariableOp-^sequential_25/Conv2D-3/Conv2D/ReadVariableOp2^sequential_25/Output-Layer/BiasAdd/ReadVariableOp1^sequential_25/Output-Layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????'
: : : : : : : : : : : : : : : : : 2v
9sequential_25/Btch_Norm-1/FusedBatchNormV3/ReadVariableOp9sequential_25/Btch_Norm-1/FusedBatchNormV3/ReadVariableOp2z
;sequential_25/Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1;sequential_25/Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_12T
(sequential_25/Btch_Norm-1/ReadVariableOp(sequential_25/Btch_Norm-1/ReadVariableOp2X
*sequential_25/Btch_Norm-1/ReadVariableOp_1*sequential_25/Btch_Norm-1/ReadVariableOp_12v
9sequential_25/Btch_Norm-2/FusedBatchNormV3/ReadVariableOp9sequential_25/Btch_Norm-2/FusedBatchNormV3/ReadVariableOp2z
;sequential_25/Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1;sequential_25/Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_12T
(sequential_25/Btch_Norm-2/ReadVariableOp(sequential_25/Btch_Norm-2/ReadVariableOp2X
*sequential_25/Btch_Norm-2/ReadVariableOp_1*sequential_25/Btch_Norm-2/ReadVariableOp_12v
9sequential_25/Btch_Norm-3/FusedBatchNormV3/ReadVariableOp9sequential_25/Btch_Norm-3/FusedBatchNormV3/ReadVariableOp2z
;sequential_25/Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1;sequential_25/Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_12T
(sequential_25/Btch_Norm-3/ReadVariableOp(sequential_25/Btch_Norm-3/ReadVariableOp2X
*sequential_25/Btch_Norm-3/ReadVariableOp_1*sequential_25/Btch_Norm-3/ReadVariableOp_12\
,sequential_25/Conv2D-1/Conv2D/ReadVariableOp,sequential_25/Conv2D-1/Conv2D/ReadVariableOp2\
,sequential_25/Conv2D-2/Conv2D/ReadVariableOp,sequential_25/Conv2D-2/Conv2D/ReadVariableOp2\
,sequential_25/Conv2D-3/Conv2D/ReadVariableOp,sequential_25/Conv2D-3/Conv2D/ReadVariableOp2f
1sequential_25/Output-Layer/BiasAdd/ReadVariableOp1sequential_25/Output-Layer/BiasAdd/ReadVariableOp2d
0sequential_25/Output-Layer/MatMul/ReadVariableOp0sequential_25/Output-Layer/MatMul/ReadVariableOp:_ [
/
_output_shapes
:?????????'

(
_user_specified_nameConv2D-1_input
?
`
D__inference_re_lu_50_layer_call_and_return_conditional_losses_128870

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????&b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????&"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????&:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?	
?
H__inference_Output-Layer_layer_call_and_return_conditional_losses_127672

inputs0
matmul_readvariableop_resource:&-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:&*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????&: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
,__inference_Btch_Norm-2_layer_call_fn_128601

inputs
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_127394?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_129232
file_prefix:
 assignvariableop_conv2d_1_kernel:&2
$assignvariableop_1_btch_norm_1_gamma:&1
#assignvariableop_2_btch_norm_1_beta:&8
*assignvariableop_3_btch_norm_1_moving_mean:&<
.assignvariableop_4_btch_norm_1_moving_variance:&<
"assignvariableop_5_conv2d_2_kernel:&&2
$assignvariableop_6_btch_norm_2_gamma:&1
#assignvariableop_7_btch_norm_2_beta:&8
*assignvariableop_8_btch_norm_2_moving_mean:&<
.assignvariableop_9_btch_norm_2_moving_variance:&=
#assignvariableop_10_conv2d_3_kernel:&&3
%assignvariableop_11_btch_norm_3_gamma:&2
$assignvariableop_12_btch_norm_3_beta:&9
+assignvariableop_13_btch_norm_3_moving_mean:&=
/assignvariableop_14_btch_norm_3_moving_variance:&9
'assignvariableop_15_output_layer_kernel:&3
%assignvariableop_16_output_layer_bias:'
assignvariableop_17_adam_iter:	 )
assignvariableop_18_adam_beta_1: )
assignvariableop_19_adam_beta_2: (
assignvariableop_20_adam_decay: 0
&assignvariableop_21_adam_learning_rate: #
assignvariableop_22_total: #
assignvariableop_23_count: %
assignvariableop_24_total_1: %
assignvariableop_25_count_1: D
*assignvariableop_26_adam_conv2d_1_kernel_m:&:
,assignvariableop_27_adam_btch_norm_1_gamma_m:&9
+assignvariableop_28_adam_btch_norm_1_beta_m:&D
*assignvariableop_29_adam_conv2d_2_kernel_m:&&:
,assignvariableop_30_adam_btch_norm_2_gamma_m:&9
+assignvariableop_31_adam_btch_norm_2_beta_m:&D
*assignvariableop_32_adam_conv2d_3_kernel_m:&&:
,assignvariableop_33_adam_btch_norm_3_gamma_m:&9
+assignvariableop_34_adam_btch_norm_3_beta_m:&@
.assignvariableop_35_adam_output_layer_kernel_m:&:
,assignvariableop_36_adam_output_layer_bias_m:D
*assignvariableop_37_adam_conv2d_1_kernel_v:&:
,assignvariableop_38_adam_btch_norm_1_gamma_v:&9
+assignvariableop_39_adam_btch_norm_1_beta_v:&D
*assignvariableop_40_adam_conv2d_2_kernel_v:&&:
,assignvariableop_41_adam_btch_norm_2_gamma_v:&9
+assignvariableop_42_adam_btch_norm_2_beta_v:&D
*assignvariableop_43_adam_conv2d_3_kernel_v:&&:
,assignvariableop_44_adam_btch_norm_3_gamma_v:&9
+assignvariableop_45_adam_btch_norm_3_beta_v:&@
.assignvariableop_46_adam_output_layer_kernel_v:&:
,assignvariableop_47_adam_output_layer_bias_v:
identity_49??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*?
value?B?1B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes5
321	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp$assignvariableop_1_btch_norm_1_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_btch_norm_1_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp*assignvariableop_3_btch_norm_1_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp.assignvariableop_4_btch_norm_1_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_2_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp$assignvariableop_6_btch_norm_2_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp#assignvariableop_7_btch_norm_2_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp*assignvariableop_8_btch_norm_2_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_btch_norm_2_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp%assignvariableop_11_btch_norm_3_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_btch_norm_3_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp+assignvariableop_13_btch_norm_3_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_btch_norm_3_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_output_layer_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp%assignvariableop_16_output_layer_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_iterIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_beta_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_decayIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp&assignvariableop_21_adam_learning_rateIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp*assignvariableop_26_adam_conv2d_1_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp,assignvariableop_27_adam_btch_norm_1_gamma_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp+assignvariableop_28_adam_btch_norm_1_beta_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_2_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp,assignvariableop_30_adam_btch_norm_2_gamma_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_btch_norm_2_beta_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp*assignvariableop_32_adam_conv2d_3_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_btch_norm_3_gamma_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp+assignvariableop_34_adam_btch_norm_3_beta_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp.assignvariableop_35_adam_output_layer_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp,assignvariableop_36_adam_output_layer_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp,assignvariableop_38_adam_btch_norm_1_gamma_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_btch_norm_1_beta_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_conv2d_2_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_btch_norm_2_gamma_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp+assignvariableop_42_adam_btch_norm_2_beta_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_3_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp,assignvariableop_44_adam_btch_norm_3_gamma_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_btch_norm_3_beta_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp.assignvariableop_46_adam_output_layer_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp,assignvariableop_47_adam_output_layer_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_48Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_49IdentityIdentity_48:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_49Identity_49:output:0*u
_input_shapesd
b: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472(
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
?
?
)__inference_Conv2D-3_layer_call_fn_128729

inputs!
unknown:&&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-3_layer_call_and_return_conditional_losses_127617w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????&: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_127593

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????&?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
`
D__inference_re_lu_48_layer_call_and_return_conditional_losses_128574

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????&b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????&"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????&:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
D__inference_Conv2D-3_layer_call_and_return_conditional_losses_127617

inputs8
conv2d_readvariableop_resource:&&
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:&&*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????&^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????&: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_127548

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????&?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
)__inference_Conv2D-2_layer_call_fn_128581

inputs!
unknown:&&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-2_layer_call_and_return_conditional_losses_127572w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????&: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
`
D__inference_re_lu_48_layer_call_and_return_conditional_losses_127563

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????&b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????&"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????&:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
,__inference_Btch_Norm-3_layer_call_fn_128788

inputs
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_127770w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
,__inference_Btch_Norm-2_layer_call_fn_128614

inputs
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_127425?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_127489

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o?:*
exponential_avg_factor%fff??
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
?
.__inference_sequential_25_layer_call_fn_128071
conv2d_1_input!
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
	unknown_3:&#
	unknown_4:&&
	unknown_5:&
	unknown_6:&
	unknown_7:&
	unknown_8:&#
	unknown_9:&&

unknown_10:&

unknown_11:&

unknown_12:&

unknown_13:&

unknown_14:&

unknown_15:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_25_layer_call_and_return_conditional_losses_127995o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????'
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????'

(
_user_specified_nameConv2D-1_input
?
?
,__inference_Btch_Norm-3_layer_call_fn_128775

inputs
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_127638w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_128842

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????&?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
r
V__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_127660

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      g
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????&U
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????&"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????&:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_128694

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????&?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_127458

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
?
-__inference_Output-Layer_layer_call_fn_128901

inputs
unknown:&
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_Output-Layer_layer_call_and_return_conditional_losses_127672o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????&: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_128860

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
exponential_avg_factor%fff??
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????&?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?Y
?
I__inference_sequential_25_layer_call_and_return_conditional_losses_128426

inputsA
'conv2d_1_conv2d_readvariableop_resource:&1
#btch_norm_1_readvariableop_resource:&3
%btch_norm_1_readvariableop_1_resource:&B
4btch_norm_1_fusedbatchnormv3_readvariableop_resource:&D
6btch_norm_1_fusedbatchnormv3_readvariableop_1_resource:&A
'conv2d_2_conv2d_readvariableop_resource:&&1
#btch_norm_2_readvariableop_resource:&3
%btch_norm_2_readvariableop_1_resource:&B
4btch_norm_2_fusedbatchnormv3_readvariableop_resource:&D
6btch_norm_2_fusedbatchnormv3_readvariableop_1_resource:&A
'conv2d_3_conv2d_readvariableop_resource:&&1
#btch_norm_3_readvariableop_resource:&3
%btch_norm_3_readvariableop_1_resource:&B
4btch_norm_3_fusedbatchnormv3_readvariableop_resource:&D
6btch_norm_3_fusedbatchnormv3_readvariableop_1_resource:&=
+output_layer_matmul_readvariableop_resource:&:
,output_layer_biasadd_readvariableop_resource:
identity??Btch_Norm-1/AssignNewValue?Btch_Norm-1/AssignNewValue_1?+Btch_Norm-1/FusedBatchNormV3/ReadVariableOp?-Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1?Btch_Norm-1/ReadVariableOp?Btch_Norm-1/ReadVariableOp_1?Btch_Norm-2/AssignNewValue?Btch_Norm-2/AssignNewValue_1?+Btch_Norm-2/FusedBatchNormV3/ReadVariableOp?-Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1?Btch_Norm-2/ReadVariableOp?Btch_Norm-2/ReadVariableOp_1?Btch_Norm-3/AssignNewValue?Btch_Norm-3/AssignNewValue_1?+Btch_Norm-3/FusedBatchNormV3/ReadVariableOp?-Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1?Btch_Norm-3/ReadVariableOp?Btch_Norm-3/ReadVariableOp_1?Conv2D-1/Conv2D/ReadVariableOp?Conv2D-2/Conv2D/ReadVariableOp?Conv2D-3/Conv2D/ReadVariableOp?#Output-Layer/BiasAdd/ReadVariableOp?"Output-Layer/MatMul/ReadVariableOp?
Conv2D-1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:&*
dtype0?
Conv2D-1/Conv2DConv2Dinputs&Conv2D-1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
z
Btch_Norm-1/ReadVariableOpReadVariableOp#btch_norm_1_readvariableop_resource*
_output_shapes
:&*
dtype0~
Btch_Norm-1/ReadVariableOp_1ReadVariableOp%btch_norm_1_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
+Btch_Norm-1/FusedBatchNormV3/ReadVariableOpReadVariableOp4btch_norm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
-Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6btch_norm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
Btch_Norm-1/FusedBatchNormV3FusedBatchNormV3Conv2D-1/Conv2D:output:0"Btch_Norm-1/ReadVariableOp:value:0$Btch_Norm-1/ReadVariableOp_1:value:03Btch_Norm-1/FusedBatchNormV3/ReadVariableOp:value:05Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
exponential_avg_factor%fff??
Btch_Norm-1/AssignNewValueAssignVariableOp4btch_norm_1_fusedbatchnormv3_readvariableop_resource)Btch_Norm-1/FusedBatchNormV3:batch_mean:0,^Btch_Norm-1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
Btch_Norm-1/AssignNewValue_1AssignVariableOp6btch_norm_1_fusedbatchnormv3_readvariableop_1_resource-Btch_Norm-1/FusedBatchNormV3:batch_variance:0.^Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0q
re_lu_48/ReluRelu Btch_Norm-1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&?
Conv2D-2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:&&*
dtype0?
Conv2D-2/Conv2DConv2Dre_lu_48/Relu:activations:0&Conv2D-2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
z
Btch_Norm-2/ReadVariableOpReadVariableOp#btch_norm_2_readvariableop_resource*
_output_shapes
:&*
dtype0~
Btch_Norm-2/ReadVariableOp_1ReadVariableOp%btch_norm_2_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
+Btch_Norm-2/FusedBatchNormV3/ReadVariableOpReadVariableOp4btch_norm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
-Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6btch_norm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
Btch_Norm-2/FusedBatchNormV3FusedBatchNormV3Conv2D-2/Conv2D:output:0"Btch_Norm-2/ReadVariableOp:value:0$Btch_Norm-2/ReadVariableOp_1:value:03Btch_Norm-2/FusedBatchNormV3/ReadVariableOp:value:05Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
exponential_avg_factor%fff??
Btch_Norm-2/AssignNewValueAssignVariableOp4btch_norm_2_fusedbatchnormv3_readvariableop_resource)Btch_Norm-2/FusedBatchNormV3:batch_mean:0,^Btch_Norm-2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
Btch_Norm-2/AssignNewValue_1AssignVariableOp6btch_norm_2_fusedbatchnormv3_readvariableop_1_resource-Btch_Norm-2/FusedBatchNormV3:batch_variance:0.^Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0q
re_lu_49/ReluRelu Btch_Norm-2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&?
Conv2D-3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:&&*
dtype0?
Conv2D-3/Conv2DConv2Dre_lu_49/Relu:activations:0&Conv2D-3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
z
Btch_Norm-3/ReadVariableOpReadVariableOp#btch_norm_3_readvariableop_resource*
_output_shapes
:&*
dtype0~
Btch_Norm-3/ReadVariableOp_1ReadVariableOp%btch_norm_3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
+Btch_Norm-3/FusedBatchNormV3/ReadVariableOpReadVariableOp4btch_norm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
-Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6btch_norm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
Btch_Norm-3/FusedBatchNormV3FusedBatchNormV3Conv2D-3/Conv2D:output:0"Btch_Norm-3/ReadVariableOp:value:0$Btch_Norm-3/ReadVariableOp_1:value:03Btch_Norm-3/FusedBatchNormV3/ReadVariableOp:value:05Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
exponential_avg_factor%fff??
Btch_Norm-3/AssignNewValueAssignVariableOp4btch_norm_3_fusedbatchnormv3_readvariableop_resource)Btch_Norm-3/FusedBatchNormV3:batch_mean:0,^Btch_Norm-3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
Btch_Norm-3/AssignNewValue_1AssignVariableOp6btch_norm_3_fusedbatchnormv3_readvariableop_1_resource-Btch_Norm-3/FusedBatchNormV3:batch_variance:0.^Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0q
re_lu_50/ReluRelu Btch_Norm-3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&?
1GlobalAveragePooling-Layer/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
GlobalAveragePooling-Layer/MeanMeanre_lu_50/Relu:activations:0:GlobalAveragePooling-Layer/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????&?
"Output-Layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:&*
dtype0?
Output-Layer/MatMulMatMul(GlobalAveragePooling-Layer/Mean:output:0*Output-Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#Output-Layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Output-Layer/BiasAddBiasAddOutput-Layer/MatMul:product:0+Output-Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????l
IdentityIdentityOutput-Layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp^Btch_Norm-1/AssignNewValue^Btch_Norm-1/AssignNewValue_1,^Btch_Norm-1/FusedBatchNormV3/ReadVariableOp.^Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1^Btch_Norm-1/ReadVariableOp^Btch_Norm-1/ReadVariableOp_1^Btch_Norm-2/AssignNewValue^Btch_Norm-2/AssignNewValue_1,^Btch_Norm-2/FusedBatchNormV3/ReadVariableOp.^Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1^Btch_Norm-2/ReadVariableOp^Btch_Norm-2/ReadVariableOp_1^Btch_Norm-3/AssignNewValue^Btch_Norm-3/AssignNewValue_1,^Btch_Norm-3/FusedBatchNormV3/ReadVariableOp.^Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1^Btch_Norm-3/ReadVariableOp^Btch_Norm-3/ReadVariableOp_1^Conv2D-1/Conv2D/ReadVariableOp^Conv2D-2/Conv2D/ReadVariableOp^Conv2D-3/Conv2D/ReadVariableOp$^Output-Layer/BiasAdd/ReadVariableOp#^Output-Layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????'
: : : : : : : : : : : : : : : : : 28
Btch_Norm-1/AssignNewValueBtch_Norm-1/AssignNewValue2<
Btch_Norm-1/AssignNewValue_1Btch_Norm-1/AssignNewValue_12Z
+Btch_Norm-1/FusedBatchNormV3/ReadVariableOp+Btch_Norm-1/FusedBatchNormV3/ReadVariableOp2^
-Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1-Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_128
Btch_Norm-1/ReadVariableOpBtch_Norm-1/ReadVariableOp2<
Btch_Norm-1/ReadVariableOp_1Btch_Norm-1/ReadVariableOp_128
Btch_Norm-2/AssignNewValueBtch_Norm-2/AssignNewValue2<
Btch_Norm-2/AssignNewValue_1Btch_Norm-2/AssignNewValue_12Z
+Btch_Norm-2/FusedBatchNormV3/ReadVariableOp+Btch_Norm-2/FusedBatchNormV3/ReadVariableOp2^
-Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1-Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_128
Btch_Norm-2/ReadVariableOpBtch_Norm-2/ReadVariableOp2<
Btch_Norm-2/ReadVariableOp_1Btch_Norm-2/ReadVariableOp_128
Btch_Norm-3/AssignNewValueBtch_Norm-3/AssignNewValue2<
Btch_Norm-3/AssignNewValue_1Btch_Norm-3/AssignNewValue_12Z
+Btch_Norm-3/FusedBatchNormV3/ReadVariableOp+Btch_Norm-3/FusedBatchNormV3/ReadVariableOp2^
-Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1-Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_128
Btch_Norm-3/ReadVariableOpBtch_Norm-3/ReadVariableOp2<
Btch_Norm-3/ReadVariableOp_1Btch_Norm-3/ReadVariableOp_12@
Conv2D-1/Conv2D/ReadVariableOpConv2D-1/Conv2D/ReadVariableOp2@
Conv2D-2/Conv2D/ReadVariableOpConv2D-2/Conv2D/ReadVariableOp2@
Conv2D-3/Conv2D/ReadVariableOpConv2D-3/Conv2D/ReadVariableOp2J
#Output-Layer/BiasAdd/ReadVariableOp#Output-Layer/BiasAdd/ReadVariableOp2H
"Output-Layer/MatMul/ReadVariableOp"Output-Layer/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????'

 
_user_specified_nameinputs
?
W
;__inference_GlobalAveragePooling-Layer_layer_call_fn_128875

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_127510i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?3
?
I__inference_sequential_25_layer_call_and_return_conditional_losses_127995

inputs)
conv2d_1_127949:& 
btch_norm_1_127952:& 
btch_norm_1_127954:& 
btch_norm_1_127956:& 
btch_norm_1_127958:&)
conv2d_2_127962:&& 
btch_norm_2_127965:& 
btch_norm_2_127967:& 
btch_norm_2_127969:& 
btch_norm_2_127971:&)
conv2d_3_127975:&& 
btch_norm_3_127978:& 
btch_norm_3_127980:& 
btch_norm_3_127982:& 
btch_norm_3_127984:&%
output_layer_127989:&!
output_layer_127991:
identity??#Btch_Norm-1/StatefulPartitionedCall?#Btch_Norm-2/StatefulPartitionedCall?#Btch_Norm-3/StatefulPartitionedCall? Conv2D-1/StatefulPartitionedCall? Conv2D-2/StatefulPartitionedCall? Conv2D-3/StatefulPartitionedCall?$Output-Layer/StatefulPartitionedCall?
 Conv2D-1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_127949*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-1_layer_call_and_return_conditional_losses_127527?
#Btch_Norm-1/StatefulPartitionedCallStatefulPartitionedCall)Conv2D-1/StatefulPartitionedCall:output:0btch_norm_1_127952btch_norm_1_127954btch_norm_1_127956btch_norm_1_127958*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_127886?
re_lu_48/PartitionedCallPartitionedCall,Btch_Norm-1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_48_layer_call_and_return_conditional_losses_127563?
 Conv2D-2/StatefulPartitionedCallStatefulPartitionedCall!re_lu_48/PartitionedCall:output:0conv2d_2_127962*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-2_layer_call_and_return_conditional_losses_127572?
#Btch_Norm-2/StatefulPartitionedCallStatefulPartitionedCall)Conv2D-2/StatefulPartitionedCall:output:0btch_norm_2_127965btch_norm_2_127967btch_norm_2_127969btch_norm_2_127971*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_127828?
re_lu_49/PartitionedCallPartitionedCall,Btch_Norm-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_49_layer_call_and_return_conditional_losses_127608?
 Conv2D-3/StatefulPartitionedCallStatefulPartitionedCall!re_lu_49/PartitionedCall:output:0conv2d_3_127975*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-3_layer_call_and_return_conditional_losses_127617?
#Btch_Norm-3/StatefulPartitionedCallStatefulPartitionedCall)Conv2D-3/StatefulPartitionedCall:output:0btch_norm_3_127978btch_norm_3_127980btch_norm_3_127982btch_norm_3_127984*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_127770?
re_lu_50/PartitionedCallPartitionedCall,Btch_Norm-3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_50_layer_call_and_return_conditional_losses_127653?
*GlobalAveragePooling-Layer/PartitionedCallPartitionedCall!re_lu_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_127660?
$Output-Layer/StatefulPartitionedCallStatefulPartitionedCall3GlobalAveragePooling-Layer/PartitionedCall:output:0output_layer_127989output_layer_127991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_Output-Layer_layer_call_and_return_conditional_losses_127672|
IdentityIdentity-Output-Layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp$^Btch_Norm-1/StatefulPartitionedCall$^Btch_Norm-2/StatefulPartitionedCall$^Btch_Norm-3/StatefulPartitionedCall!^Conv2D-1/StatefulPartitionedCall!^Conv2D-2/StatefulPartitionedCall!^Conv2D-3/StatefulPartitionedCall%^Output-Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????'
: : : : : : : : : : : : : : : : : 2J
#Btch_Norm-1/StatefulPartitionedCall#Btch_Norm-1/StatefulPartitionedCall2J
#Btch_Norm-2/StatefulPartitionedCall#Btch_Norm-2/StatefulPartitionedCall2J
#Btch_Norm-3/StatefulPartitionedCall#Btch_Norm-3/StatefulPartitionedCall2D
 Conv2D-1/StatefulPartitionedCall Conv2D-1/StatefulPartitionedCall2D
 Conv2D-2/StatefulPartitionedCall Conv2D-2/StatefulPartitionedCall2D
 Conv2D-3/StatefulPartitionedCall Conv2D-3/StatefulPartitionedCall2L
$Output-Layer/StatefulPartitionedCall$Output-Layer/StatefulPartitionedCall:W S
/
_output_shapes
:?????????'

 
_user_specified_nameinputs
?
?
,__inference_Btch_Norm-1_layer_call_fn_128479

inputs
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_127548w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_128658

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
r
V__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_128886

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_re_lu_49_layer_call_and_return_conditional_losses_127608

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????&b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????&"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????&:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
D__inference_Conv2D-2_layer_call_and_return_conditional_losses_128588

inputs8
conv2d_readvariableop_resource:&&
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:&&*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????&^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????&: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
E
)__inference_re_lu_49_layer_call_fn_128717

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_49_layer_call_and_return_conditional_losses_127608h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????&"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????&:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?3
?
I__inference_sequential_25_layer_call_and_return_conditional_losses_128169
conv2d_1_input)
conv2d_1_128123:& 
btch_norm_1_128126:& 
btch_norm_1_128128:& 
btch_norm_1_128130:& 
btch_norm_1_128132:&)
conv2d_2_128136:&& 
btch_norm_2_128139:& 
btch_norm_2_128141:& 
btch_norm_2_128143:& 
btch_norm_2_128145:&)
conv2d_3_128149:&& 
btch_norm_3_128152:& 
btch_norm_3_128154:& 
btch_norm_3_128156:& 
btch_norm_3_128158:&%
output_layer_128163:&!
output_layer_128165:
identity??#Btch_Norm-1/StatefulPartitionedCall?#Btch_Norm-2/StatefulPartitionedCall?#Btch_Norm-3/StatefulPartitionedCall? Conv2D-1/StatefulPartitionedCall? Conv2D-2/StatefulPartitionedCall? Conv2D-3/StatefulPartitionedCall?$Output-Layer/StatefulPartitionedCall?
 Conv2D-1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_128123*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-1_layer_call_and_return_conditional_losses_127527?
#Btch_Norm-1/StatefulPartitionedCallStatefulPartitionedCall)Conv2D-1/StatefulPartitionedCall:output:0btch_norm_1_128126btch_norm_1_128128btch_norm_1_128130btch_norm_1_128132*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_127886?
re_lu_48/PartitionedCallPartitionedCall,Btch_Norm-1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_48_layer_call_and_return_conditional_losses_127563?
 Conv2D-2/StatefulPartitionedCallStatefulPartitionedCall!re_lu_48/PartitionedCall:output:0conv2d_2_128136*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-2_layer_call_and_return_conditional_losses_127572?
#Btch_Norm-2/StatefulPartitionedCallStatefulPartitionedCall)Conv2D-2/StatefulPartitionedCall:output:0btch_norm_2_128139btch_norm_2_128141btch_norm_2_128143btch_norm_2_128145*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_127828?
re_lu_49/PartitionedCallPartitionedCall,Btch_Norm-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_49_layer_call_and_return_conditional_losses_127608?
 Conv2D-3/StatefulPartitionedCallStatefulPartitionedCall!re_lu_49/PartitionedCall:output:0conv2d_3_128149*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-3_layer_call_and_return_conditional_losses_127617?
#Btch_Norm-3/StatefulPartitionedCallStatefulPartitionedCall)Conv2D-3/StatefulPartitionedCall:output:0btch_norm_3_128152btch_norm_3_128154btch_norm_3_128156btch_norm_3_128158*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_127770?
re_lu_50/PartitionedCallPartitionedCall,Btch_Norm-3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_50_layer_call_and_return_conditional_losses_127653?
*GlobalAveragePooling-Layer/PartitionedCallPartitionedCall!re_lu_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_127660?
$Output-Layer/StatefulPartitionedCallStatefulPartitionedCall3GlobalAveragePooling-Layer/PartitionedCall:output:0output_layer_128163output_layer_128165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_Output-Layer_layer_call_and_return_conditional_losses_127672|
IdentityIdentity-Output-Layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp$^Btch_Norm-1/StatefulPartitionedCall$^Btch_Norm-2/StatefulPartitionedCall$^Btch_Norm-3/StatefulPartitionedCall!^Conv2D-1/StatefulPartitionedCall!^Conv2D-2/StatefulPartitionedCall!^Conv2D-3/StatefulPartitionedCall%^Output-Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????'
: : : : : : : : : : : : : : : : : 2J
#Btch_Norm-1/StatefulPartitionedCall#Btch_Norm-1/StatefulPartitionedCall2J
#Btch_Norm-2/StatefulPartitionedCall#Btch_Norm-2/StatefulPartitionedCall2J
#Btch_Norm-3/StatefulPartitionedCall#Btch_Norm-3/StatefulPartitionedCall2D
 Conv2D-1/StatefulPartitionedCall Conv2D-1/StatefulPartitionedCall2D
 Conv2D-2/StatefulPartitionedCall Conv2D-2/StatefulPartitionedCall2D
 Conv2D-3/StatefulPartitionedCall Conv2D-3/StatefulPartitionedCall2L
$Output-Layer/StatefulPartitionedCall$Output-Layer/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????'

(
_user_specified_nameConv2D-1_input
?
?
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_127770

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
exponential_avg_factor%fff??
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????&?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?H
?
I__inference_sequential_25_layer_call_and_return_conditional_losses_128360

inputsA
'conv2d_1_conv2d_readvariableop_resource:&1
#btch_norm_1_readvariableop_resource:&3
%btch_norm_1_readvariableop_1_resource:&B
4btch_norm_1_fusedbatchnormv3_readvariableop_resource:&D
6btch_norm_1_fusedbatchnormv3_readvariableop_1_resource:&A
'conv2d_2_conv2d_readvariableop_resource:&&1
#btch_norm_2_readvariableop_resource:&3
%btch_norm_2_readvariableop_1_resource:&B
4btch_norm_2_fusedbatchnormv3_readvariableop_resource:&D
6btch_norm_2_fusedbatchnormv3_readvariableop_1_resource:&A
'conv2d_3_conv2d_readvariableop_resource:&&1
#btch_norm_3_readvariableop_resource:&3
%btch_norm_3_readvariableop_1_resource:&B
4btch_norm_3_fusedbatchnormv3_readvariableop_resource:&D
6btch_norm_3_fusedbatchnormv3_readvariableop_1_resource:&=
+output_layer_matmul_readvariableop_resource:&:
,output_layer_biasadd_readvariableop_resource:
identity??+Btch_Norm-1/FusedBatchNormV3/ReadVariableOp?-Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1?Btch_Norm-1/ReadVariableOp?Btch_Norm-1/ReadVariableOp_1?+Btch_Norm-2/FusedBatchNormV3/ReadVariableOp?-Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1?Btch_Norm-2/ReadVariableOp?Btch_Norm-2/ReadVariableOp_1?+Btch_Norm-3/FusedBatchNormV3/ReadVariableOp?-Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1?Btch_Norm-3/ReadVariableOp?Btch_Norm-3/ReadVariableOp_1?Conv2D-1/Conv2D/ReadVariableOp?Conv2D-2/Conv2D/ReadVariableOp?Conv2D-3/Conv2D/ReadVariableOp?#Output-Layer/BiasAdd/ReadVariableOp?"Output-Layer/MatMul/ReadVariableOp?
Conv2D-1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:&*
dtype0?
Conv2D-1/Conv2DConv2Dinputs&Conv2D-1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
z
Btch_Norm-1/ReadVariableOpReadVariableOp#btch_norm_1_readvariableop_resource*
_output_shapes
:&*
dtype0~
Btch_Norm-1/ReadVariableOp_1ReadVariableOp%btch_norm_1_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
+Btch_Norm-1/FusedBatchNormV3/ReadVariableOpReadVariableOp4btch_norm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
-Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6btch_norm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
Btch_Norm-1/FusedBatchNormV3FusedBatchNormV3Conv2D-1/Conv2D:output:0"Btch_Norm-1/ReadVariableOp:value:0$Btch_Norm-1/ReadVariableOp_1:value:03Btch_Norm-1/FusedBatchNormV3/ReadVariableOp:value:05Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
is_training( q
re_lu_48/ReluRelu Btch_Norm-1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&?
Conv2D-2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:&&*
dtype0?
Conv2D-2/Conv2DConv2Dre_lu_48/Relu:activations:0&Conv2D-2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
z
Btch_Norm-2/ReadVariableOpReadVariableOp#btch_norm_2_readvariableop_resource*
_output_shapes
:&*
dtype0~
Btch_Norm-2/ReadVariableOp_1ReadVariableOp%btch_norm_2_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
+Btch_Norm-2/FusedBatchNormV3/ReadVariableOpReadVariableOp4btch_norm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
-Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6btch_norm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
Btch_Norm-2/FusedBatchNormV3FusedBatchNormV3Conv2D-2/Conv2D:output:0"Btch_Norm-2/ReadVariableOp:value:0$Btch_Norm-2/ReadVariableOp_1:value:03Btch_Norm-2/FusedBatchNormV3/ReadVariableOp:value:05Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
is_training( q
re_lu_49/ReluRelu Btch_Norm-2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&?
Conv2D-3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:&&*
dtype0?
Conv2D-3/Conv2DConv2Dre_lu_49/Relu:activations:0&Conv2D-3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
z
Btch_Norm-3/ReadVariableOpReadVariableOp#btch_norm_3_readvariableop_resource*
_output_shapes
:&*
dtype0~
Btch_Norm-3/ReadVariableOp_1ReadVariableOp%btch_norm_3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
+Btch_Norm-3/FusedBatchNormV3/ReadVariableOpReadVariableOp4btch_norm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
-Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6btch_norm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
Btch_Norm-3/FusedBatchNormV3FusedBatchNormV3Conv2D-3/Conv2D:output:0"Btch_Norm-3/ReadVariableOp:value:0$Btch_Norm-3/ReadVariableOp_1:value:03Btch_Norm-3/FusedBatchNormV3/ReadVariableOp:value:05Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
is_training( q
re_lu_50/ReluRelu Btch_Norm-3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&?
1GlobalAveragePooling-Layer/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
GlobalAveragePooling-Layer/MeanMeanre_lu_50/Relu:activations:0:GlobalAveragePooling-Layer/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????&?
"Output-Layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:&*
dtype0?
Output-Layer/MatMulMatMul(GlobalAveragePooling-Layer/Mean:output:0*Output-Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#Output-Layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
Output-Layer/BiasAddBiasAddOutput-Layer/MatMul:product:0+Output-Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????l
IdentityIdentityOutput-Layer/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^Btch_Norm-1/FusedBatchNormV3/ReadVariableOp.^Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1^Btch_Norm-1/ReadVariableOp^Btch_Norm-1/ReadVariableOp_1,^Btch_Norm-2/FusedBatchNormV3/ReadVariableOp.^Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1^Btch_Norm-2/ReadVariableOp^Btch_Norm-2/ReadVariableOp_1,^Btch_Norm-3/FusedBatchNormV3/ReadVariableOp.^Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1^Btch_Norm-3/ReadVariableOp^Btch_Norm-3/ReadVariableOp_1^Conv2D-1/Conv2D/ReadVariableOp^Conv2D-2/Conv2D/ReadVariableOp^Conv2D-3/Conv2D/ReadVariableOp$^Output-Layer/BiasAdd/ReadVariableOp#^Output-Layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????'
: : : : : : : : : : : : : : : : : 2Z
+Btch_Norm-1/FusedBatchNormV3/ReadVariableOp+Btch_Norm-1/FusedBatchNormV3/ReadVariableOp2^
-Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1-Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_128
Btch_Norm-1/ReadVariableOpBtch_Norm-1/ReadVariableOp2<
Btch_Norm-1/ReadVariableOp_1Btch_Norm-1/ReadVariableOp_12Z
+Btch_Norm-2/FusedBatchNormV3/ReadVariableOp+Btch_Norm-2/FusedBatchNormV3/ReadVariableOp2^
-Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1-Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_128
Btch_Norm-2/ReadVariableOpBtch_Norm-2/ReadVariableOp2<
Btch_Norm-2/ReadVariableOp_1Btch_Norm-2/ReadVariableOp_12Z
+Btch_Norm-3/FusedBatchNormV3/ReadVariableOp+Btch_Norm-3/FusedBatchNormV3/ReadVariableOp2^
-Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1-Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_128
Btch_Norm-3/ReadVariableOpBtch_Norm-3/ReadVariableOp2<
Btch_Norm-3/ReadVariableOp_1Btch_Norm-3/ReadVariableOp_12@
Conv2D-1/Conv2D/ReadVariableOpConv2D-1/Conv2D/ReadVariableOp2@
Conv2D-2/Conv2D/ReadVariableOpConv2D-2/Conv2D/ReadVariableOp2@
Conv2D-3/Conv2D/ReadVariableOpConv2D-3/Conv2D/ReadVariableOp2J
#Output-Layer/BiasAdd/ReadVariableOp#Output-Layer/BiasAdd/ReadVariableOp2H
"Output-Layer/MatMul/ReadVariableOp"Output-Layer/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????'

 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_128564

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
exponential_avg_factor%fff??
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????&?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
,__inference_Btch_Norm-1_layer_call_fn_128466

inputs
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_127361?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_127828

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
exponential_avg_factor%fff??
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????&?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_127361

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o?:*
exponential_avg_factor%fff??
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_128806

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
?
,__inference_Btch_Norm-1_layer_call_fn_128492

inputs
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_127886w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
r
V__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_128892

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      g
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????&U
IdentityIdentityMean:output:0*
T0*'
_output_shapes
:?????????&"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????&:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_127394

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
?
,__inference_Btch_Norm-3_layer_call_fn_128749

inputs
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_127458?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
?
,__inference_Btch_Norm-3_layer_call_fn_128762

inputs
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_127489?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
?
D__inference_Conv2D-2_layer_call_and_return_conditional_losses_127572

inputs8
conv2d_readvariableop_resource:&&
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:&&*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????&^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????&: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
,__inference_Btch_Norm-2_layer_call_fn_128640

inputs
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_127828w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
W
;__inference_GlobalAveragePooling-Layer_layer_call_fn_128880

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
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_127660`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????&"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????&:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?3
?
I__inference_sequential_25_layer_call_and_return_conditional_losses_127679

inputs)
conv2d_1_127528:& 
btch_norm_1_127549:& 
btch_norm_1_127551:& 
btch_norm_1_127553:& 
btch_norm_1_127555:&)
conv2d_2_127573:&& 
btch_norm_2_127594:& 
btch_norm_2_127596:& 
btch_norm_2_127598:& 
btch_norm_2_127600:&)
conv2d_3_127618:&& 
btch_norm_3_127639:& 
btch_norm_3_127641:& 
btch_norm_3_127643:& 
btch_norm_3_127645:&%
output_layer_127673:&!
output_layer_127675:
identity??#Btch_Norm-1/StatefulPartitionedCall?#Btch_Norm-2/StatefulPartitionedCall?#Btch_Norm-3/StatefulPartitionedCall? Conv2D-1/StatefulPartitionedCall? Conv2D-2/StatefulPartitionedCall? Conv2D-3/StatefulPartitionedCall?$Output-Layer/StatefulPartitionedCall?
 Conv2D-1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_127528*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-1_layer_call_and_return_conditional_losses_127527?
#Btch_Norm-1/StatefulPartitionedCallStatefulPartitionedCall)Conv2D-1/StatefulPartitionedCall:output:0btch_norm_1_127549btch_norm_1_127551btch_norm_1_127553btch_norm_1_127555*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_127548?
re_lu_48/PartitionedCallPartitionedCall,Btch_Norm-1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_48_layer_call_and_return_conditional_losses_127563?
 Conv2D-2/StatefulPartitionedCallStatefulPartitionedCall!re_lu_48/PartitionedCall:output:0conv2d_2_127573*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-2_layer_call_and_return_conditional_losses_127572?
#Btch_Norm-2/StatefulPartitionedCallStatefulPartitionedCall)Conv2D-2/StatefulPartitionedCall:output:0btch_norm_2_127594btch_norm_2_127596btch_norm_2_127598btch_norm_2_127600*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_127593?
re_lu_49/PartitionedCallPartitionedCall,Btch_Norm-2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_49_layer_call_and_return_conditional_losses_127608?
 Conv2D-3/StatefulPartitionedCallStatefulPartitionedCall!re_lu_49/PartitionedCall:output:0conv2d_3_127618*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-3_layer_call_and_return_conditional_losses_127617?
#Btch_Norm-3/StatefulPartitionedCallStatefulPartitionedCall)Conv2D-3/StatefulPartitionedCall:output:0btch_norm_3_127639btch_norm_3_127641btch_norm_3_127643btch_norm_3_127645*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_127638?
re_lu_50/PartitionedCallPartitionedCall,Btch_Norm-3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_50_layer_call_and_return_conditional_losses_127653?
*GlobalAveragePooling-Layer/PartitionedCallPartitionedCall!re_lu_50/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_127660?
$Output-Layer/StatefulPartitionedCallStatefulPartitionedCall3GlobalAveragePooling-Layer/PartitionedCall:output:0output_layer_127673output_layer_127675*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_Output-Layer_layer_call_and_return_conditional_losses_127672|
IdentityIdentity-Output-Layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp$^Btch_Norm-1/StatefulPartitionedCall$^Btch_Norm-2/StatefulPartitionedCall$^Btch_Norm-3/StatefulPartitionedCall!^Conv2D-1/StatefulPartitionedCall!^Conv2D-2/StatefulPartitionedCall!^Conv2D-3/StatefulPartitionedCall%^Output-Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????'
: : : : : : : : : : : : : : : : : 2J
#Btch_Norm-1/StatefulPartitionedCall#Btch_Norm-1/StatefulPartitionedCall2J
#Btch_Norm-2/StatefulPartitionedCall#Btch_Norm-2/StatefulPartitionedCall2J
#Btch_Norm-3/StatefulPartitionedCall#Btch_Norm-3/StatefulPartitionedCall2D
 Conv2D-1/StatefulPartitionedCall Conv2D-1/StatefulPartitionedCall2D
 Conv2D-2/StatefulPartitionedCall Conv2D-2/StatefulPartitionedCall2D
 Conv2D-3/StatefulPartitionedCall Conv2D-3/StatefulPartitionedCall2L
$Output-Layer/StatefulPartitionedCall$Output-Layer/StatefulPartitionedCall:W S
/
_output_shapes
:?????????'

 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_128676

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o?:*
exponential_avg_factor%fff??
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
?
.__inference_sequential_25_layer_call_fn_128294

inputs!
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
	unknown_3:&#
	unknown_4:&&
	unknown_5:&
	unknown_6:&
	unknown_7:&
	unknown_8:&#
	unknown_9:&&

unknown_10:&

unknown_11:&

unknown_12:&

unknown_13:&

unknown_14:&

unknown_15:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_25_layer_call_and_return_conditional_losses_127995o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????'
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????'

 
_user_specified_nameinputs
?
`
D__inference_re_lu_50_layer_call_and_return_conditional_losses_127653

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????&b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????&"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????&:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_128712

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
exponential_avg_factor%fff??
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????&?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_127886

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
exponential_avg_factor%fff??
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????&?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_127425

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o?:*
exponential_avg_factor%fff??
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_128216
conv2d_1_input!
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
	unknown_3:&#
	unknown_4:&&
	unknown_5:&
	unknown_6:&
	unknown_7:&
	unknown_8:&#
	unknown_9:&&

unknown_10:&

unknown_11:&

unknown_12:&

unknown_13:&

unknown_14:&

unknown_15:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_127308o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:?????????'
: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????'

(
_user_specified_nameConv2D-1_input
?
?
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_128510

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_128546

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o?:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????&?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????&: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
`
D__inference_re_lu_49_layer_call_and_return_conditional_losses_128722

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????&b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????&"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????&:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_128528

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o?:*
exponential_avg_factor%fff??
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
?
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_128824

inputs%
readvariableop_resource:&'
readvariableop_1_resource:&6
(fusedbatchnormv3_readvariableop_resource:&8
*fusedbatchnormv3_readvariableop_1_resource:&
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o?:*
exponential_avg_factor%fff??
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
E
)__inference_re_lu_50_layer_call_fn_128865

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_50_layer_call_and_return_conditional_losses_127653h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????&"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????&:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
?
,__inference_Btch_Norm-1_layer_call_fn_128453

inputs
unknown:&
	unknown_0:&
	unknown_1:&
	unknown_2:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_127330?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????&: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
?
?
)__inference_Conv2D-1_layer_call_fn_128433

inputs!
unknown:&
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_Conv2D-1_layer_call_and_return_conditional_losses_127527w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????&`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????'
: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????'

 
_user_specified_nameinputs
?
E
)__inference_re_lu_48_layer_call_fn_128569

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_re_lu_48_layer_call_and_return_conditional_losses_127563h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????&"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????&:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
?a
?
__inference__traced_save_129078
file_prefix.
*savev2_conv2d_1_kernel_read_readvariableop0
,savev2_btch_norm_1_gamma_read_readvariableop/
+savev2_btch_norm_1_beta_read_readvariableop6
2savev2_btch_norm_1_moving_mean_read_readvariableop:
6savev2_btch_norm_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop0
,savev2_btch_norm_2_gamma_read_readvariableop/
+savev2_btch_norm_2_beta_read_readvariableop6
2savev2_btch_norm_2_moving_mean_read_readvariableop:
6savev2_btch_norm_2_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop0
,savev2_btch_norm_3_gamma_read_readvariableop/
+savev2_btch_norm_3_beta_read_readvariableop6
2savev2_btch_norm_3_moving_mean_read_readvariableop:
6savev2_btch_norm_3_moving_variance_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop7
3savev2_adam_btch_norm_1_gamma_m_read_readvariableop6
2savev2_adam_btch_norm_1_beta_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop7
3savev2_adam_btch_norm_2_gamma_m_read_readvariableop6
2savev2_adam_btch_norm_2_beta_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop7
3savev2_adam_btch_norm_3_gamma_m_read_readvariableop6
2savev2_adam_btch_norm_3_beta_m_read_readvariableop9
5savev2_adam_output_layer_kernel_m_read_readvariableop7
3savev2_adam_output_layer_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop7
3savev2_adam_btch_norm_1_gamma_v_read_readvariableop6
2savev2_adam_btch_norm_1_beta_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop7
3savev2_adam_btch_norm_2_gamma_v_read_readvariableop6
2savev2_adam_btch_norm_2_beta_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop7
3savev2_adam_btch_norm_3_gamma_v_read_readvariableop6
2savev2_adam_btch_norm_3_beta_v_read_readvariableop9
5savev2_adam_output_layer_kernel_v_read_readvariableop7
3savev2_adam_output_layer_bias_v_read_readvariableop
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
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*?
value?B?1B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:1*
dtype0*u
valuelBj1B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_1_kernel_read_readvariableop,savev2_btch_norm_1_gamma_read_readvariableop+savev2_btch_norm_1_beta_read_readvariableop2savev2_btch_norm_1_moving_mean_read_readvariableop6savev2_btch_norm_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop,savev2_btch_norm_2_gamma_read_readvariableop+savev2_btch_norm_2_beta_read_readvariableop2savev2_btch_norm_2_moving_mean_read_readvariableop6savev2_btch_norm_2_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop,savev2_btch_norm_3_gamma_read_readvariableop+savev2_btch_norm_3_beta_read_readvariableop2savev2_btch_norm_3_moving_mean_read_readvariableop6savev2_btch_norm_3_moving_variance_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop3savev2_adam_btch_norm_1_gamma_m_read_readvariableop2savev2_adam_btch_norm_1_beta_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop3savev2_adam_btch_norm_2_gamma_m_read_readvariableop2savev2_adam_btch_norm_2_beta_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop3savev2_adam_btch_norm_3_gamma_m_read_readvariableop2savev2_adam_btch_norm_3_beta_m_read_readvariableop5savev2_adam_output_layer_kernel_m_read_readvariableop3savev2_adam_output_layer_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop3savev2_adam_btch_norm_1_gamma_v_read_readvariableop2savev2_adam_btch_norm_1_beta_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop3savev2_adam_btch_norm_2_gamma_v_read_readvariableop2savev2_adam_btch_norm_2_beta_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop3savev2_adam_btch_norm_3_gamma_v_read_readvariableop2savev2_adam_btch_norm_3_beta_v_read_readvariableop5savev2_adam_output_layer_kernel_v_read_readvariableop3savev2_adam_output_layer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes5
321	?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :&:&:&:&:&:&&:&:&:&:&:&&:&:&:&:&:&:: : : : : : : : : :&:&:&:&&:&:&:&&:&:&:&::&:&:&:&&:&:&:&&:&:&:&:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:&: 

_output_shapes
:&: 

_output_shapes
:&: 

_output_shapes
:&: 

_output_shapes
:&:,(
&
_output_shapes
:&&: 

_output_shapes
:&: 

_output_shapes
:&: 	

_output_shapes
:&: 


_output_shapes
:&:,(
&
_output_shapes
:&&: 

_output_shapes
:&: 

_output_shapes
:&: 

_output_shapes
:&: 

_output_shapes
:&:$ 

_output_shapes

:&: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:&: 

_output_shapes
:&: 

_output_shapes
:&:,(
&
_output_shapes
:&&: 

_output_shapes
:&:  

_output_shapes
:&:,!(
&
_output_shapes
:&&: "

_output_shapes
:&: #

_output_shapes
:&:$$ 

_output_shapes

:&: %

_output_shapes
::,&(
&
_output_shapes
:&: '

_output_shapes
:&: (

_output_shapes
:&:,)(
&
_output_shapes
:&&: *

_output_shapes
:&: +

_output_shapes
:&:,,(
&
_output_shapes
:&&: -

_output_shapes
:&: .

_output_shapes
:&:$/ 

_output_shapes

:&: 0

_output_shapes
::1

_output_shapes
: 
?	
?
H__inference_Output-Layer_layer_call_and_return_conditional_losses_128911

inputs0
matmul_readvariableop_resource:&-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:&*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????&: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????&
 
_user_specified_nameinputs
?
r
V__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_127510

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Q
Conv2D-1_input?
 serving_default_Conv2D-1_input:0?????????'
@
Output-Layer0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer-9
layer_with_weights-6
layer-10
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_sequential
?

kernel
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
 	variables
!trainable_variables
"regularization_losses
#	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

$kernel
%	variables
&trainable_variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
)axis
	*gamma
+beta
,moving_mean
-moving_variance
.	variables
/trainable_variables
0regularization_losses
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
2	variables
3trainable_variables
4regularization_losses
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

6kernel
7	variables
8trainable_variables
9regularization_losses
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Riter

Sbeta_1

Tbeta_2
	Udecay
Vlearning_ratem?m?m?$m?*m?+m?6m?<m?=m?Lm?Mm?v?v?v?$v?*v?+v?6v?<v?=v?Lv?Mv?"
	optimizer
?
0
1
2
3
4
$5
*6
+7
,8
-9
610
<11
=12
>13
?14
L15
M16"
trackable_list_wrapper
n
0
1
2
$3
*4
+5
66
<7
=8
L9
M10"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Wnon_trainable_variables

Xlayers
Ymetrics
Zlayer_regularization_losses
[layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
):'&2Conv2D-1/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:&2Btch_Norm-1/gamma
:&2Btch_Norm-1/beta
':%& (2Btch_Norm-1/moving_mean
+:)& (2Btch_Norm-1/moving_variance
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
 	variables
!trainable_variables
"regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'&&2Conv2D-2/kernel
'
$0"
trackable_list_wrapper
'
$0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
%	variables
&trainable_variables
'regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:&2Btch_Norm-2/gamma
:&2Btch_Norm-2/beta
':%& (2Btch_Norm-2/moving_mean
+:)& (2Btch_Norm-2/moving_variance
<
*0
+1
,2
-3"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
.	variables
/trainable_variables
0regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
2	variables
3trainable_variables
4regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'&&2Conv2D-3/kernel
'
60"
trackable_list_wrapper
'
60"
trackable_list_wrapper
 "
trackable_list_wrapper
?
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
7	variables
8trainable_variables
9regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:&2Btch_Norm-3/gamma
:&2Btch_Norm-3/beta
':%& (2Btch_Norm-3/moving_mean
+:)& (2Btch_Norm-3/moving_variance
<
<0
=1
>2
?3"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#&2Output-Layer/kernel
:2Output-Layer/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
J
0
1
,2
-3
>4
?5"
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
0
?0
?1"
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
.
0
1"
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
.
,0
-1"
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
.
>0
?1"
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
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
.:,&2Adam/Conv2D-1/kernel/m
$:"&2Adam/Btch_Norm-1/gamma/m
#:!&2Adam/Btch_Norm-1/beta/m
.:,&&2Adam/Conv2D-2/kernel/m
$:"&2Adam/Btch_Norm-2/gamma/m
#:!&2Adam/Btch_Norm-2/beta/m
.:,&&2Adam/Conv2D-3/kernel/m
$:"&2Adam/Btch_Norm-3/gamma/m
#:!&2Adam/Btch_Norm-3/beta/m
*:(&2Adam/Output-Layer/kernel/m
$:"2Adam/Output-Layer/bias/m
.:,&2Adam/Conv2D-1/kernel/v
$:"&2Adam/Btch_Norm-1/gamma/v
#:!&2Adam/Btch_Norm-1/beta/v
.:,&&2Adam/Conv2D-2/kernel/v
$:"&2Adam/Btch_Norm-2/gamma/v
#:!&2Adam/Btch_Norm-2/beta/v
.:,&&2Adam/Conv2D-3/kernel/v
$:"&2Adam/Btch_Norm-3/gamma/v
#:!&2Adam/Btch_Norm-3/beta/v
*:(&2Adam/Output-Layer/kernel/v
$:"2Adam/Output-Layer/bias/v
?2?
.__inference_sequential_25_layer_call_fn_127716
.__inference_sequential_25_layer_call_fn_128255
.__inference_sequential_25_layer_call_fn_128294
.__inference_sequential_25_layer_call_fn_128071?
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
I__inference_sequential_25_layer_call_and_return_conditional_losses_128360
I__inference_sequential_25_layer_call_and_return_conditional_losses_128426
I__inference_sequential_25_layer_call_and_return_conditional_losses_128120
I__inference_sequential_25_layer_call_and_return_conditional_losses_128169?
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
!__inference__wrapped_model_127308Conv2D-1_input"?
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
)__inference_Conv2D-1_layer_call_fn_128433?
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
D__inference_Conv2D-1_layer_call_and_return_conditional_losses_128440?
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
?2?
,__inference_Btch_Norm-1_layer_call_fn_128453
,__inference_Btch_Norm-1_layer_call_fn_128466
,__inference_Btch_Norm-1_layer_call_fn_128479
,__inference_Btch_Norm-1_layer_call_fn_128492?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_128510
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_128528
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_128546
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_128564?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_re_lu_48_layer_call_fn_128569?
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
D__inference_re_lu_48_layer_call_and_return_conditional_losses_128574?
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
)__inference_Conv2D-2_layer_call_fn_128581?
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
D__inference_Conv2D-2_layer_call_and_return_conditional_losses_128588?
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
?2?
,__inference_Btch_Norm-2_layer_call_fn_128601
,__inference_Btch_Norm-2_layer_call_fn_128614
,__inference_Btch_Norm-2_layer_call_fn_128627
,__inference_Btch_Norm-2_layer_call_fn_128640?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_128658
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_128676
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_128694
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_128712?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_re_lu_49_layer_call_fn_128717?
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
D__inference_re_lu_49_layer_call_and_return_conditional_losses_128722?
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
)__inference_Conv2D-3_layer_call_fn_128729?
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
D__inference_Conv2D-3_layer_call_and_return_conditional_losses_128736?
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
?2?
,__inference_Btch_Norm-3_layer_call_fn_128749
,__inference_Btch_Norm-3_layer_call_fn_128762
,__inference_Btch_Norm-3_layer_call_fn_128775
,__inference_Btch_Norm-3_layer_call_fn_128788?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_128806
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_128824
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_128842
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_128860?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_re_lu_50_layer_call_fn_128865?
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
D__inference_re_lu_50_layer_call_and_return_conditional_losses_128870?
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
?2?
;__inference_GlobalAveragePooling-Layer_layer_call_fn_128875
;__inference_GlobalAveragePooling-Layer_layer_call_fn_128880?
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
?2?
V__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_128886
V__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_128892?
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
-__inference_Output-Layer_layer_call_fn_128901?
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
H__inference_Output-Layer_layer_call_and_return_conditional_losses_128911?
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
$__inference_signature_wrapper_128216Conv2D-1_input"?
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
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_128510?M?J
C?@
:?7
inputs+???????????????????????????&
p 
? "??<
5?2
0+???????????????????????????&
? ?
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_128528?M?J
C?@
:?7
inputs+???????????????????????????&
p
? "??<
5?2
0+???????????????????????????&
? ?
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_128546r;?8
1?.
(?%
inputs?????????&
p 
? "-?*
#? 
0?????????&
? ?
G__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_128564r;?8
1?.
(?%
inputs?????????&
p
? "-?*
#? 
0?????????&
? ?
,__inference_Btch_Norm-1_layer_call_fn_128453?M?J
C?@
:?7
inputs+???????????????????????????&
p 
? "2?/+???????????????????????????&?
,__inference_Btch_Norm-1_layer_call_fn_128466?M?J
C?@
:?7
inputs+???????????????????????????&
p
? "2?/+???????????????????????????&?
,__inference_Btch_Norm-1_layer_call_fn_128479e;?8
1?.
(?%
inputs?????????&
p 
? " ??????????&?
,__inference_Btch_Norm-1_layer_call_fn_128492e;?8
1?.
(?%
inputs?????????&
p
? " ??????????&?
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_128658?*+,-M?J
C?@
:?7
inputs+???????????????????????????&
p 
? "??<
5?2
0+???????????????????????????&
? ?
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_128676?*+,-M?J
C?@
:?7
inputs+???????????????????????????&
p
? "??<
5?2
0+???????????????????????????&
? ?
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_128694r*+,-;?8
1?.
(?%
inputs?????????&
p 
? "-?*
#? 
0?????????&
? ?
G__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_128712r*+,-;?8
1?.
(?%
inputs?????????&
p
? "-?*
#? 
0?????????&
? ?
,__inference_Btch_Norm-2_layer_call_fn_128601?*+,-M?J
C?@
:?7
inputs+???????????????????????????&
p 
? "2?/+???????????????????????????&?
,__inference_Btch_Norm-2_layer_call_fn_128614?*+,-M?J
C?@
:?7
inputs+???????????????????????????&
p
? "2?/+???????????????????????????&?
,__inference_Btch_Norm-2_layer_call_fn_128627e*+,-;?8
1?.
(?%
inputs?????????&
p 
? " ??????????&?
,__inference_Btch_Norm-2_layer_call_fn_128640e*+,-;?8
1?.
(?%
inputs?????????&
p
? " ??????????&?
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_128806?<=>?M?J
C?@
:?7
inputs+???????????????????????????&
p 
? "??<
5?2
0+???????????????????????????&
? ?
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_128824?<=>?M?J
C?@
:?7
inputs+???????????????????????????&
p
? "??<
5?2
0+???????????????????????????&
? ?
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_128842r<=>?;?8
1?.
(?%
inputs?????????&
p 
? "-?*
#? 
0?????????&
? ?
G__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_128860r<=>?;?8
1?.
(?%
inputs?????????&
p
? "-?*
#? 
0?????????&
? ?
,__inference_Btch_Norm-3_layer_call_fn_128749?<=>?M?J
C?@
:?7
inputs+???????????????????????????&
p 
? "2?/+???????????????????????????&?
,__inference_Btch_Norm-3_layer_call_fn_128762?<=>?M?J
C?@
:?7
inputs+???????????????????????????&
p
? "2?/+???????????????????????????&?
,__inference_Btch_Norm-3_layer_call_fn_128775e<=>?;?8
1?.
(?%
inputs?????????&
p 
? " ??????????&?
,__inference_Btch_Norm-3_layer_call_fn_128788e<=>?;?8
1?.
(?%
inputs?????????&
p
? " ??????????&?
D__inference_Conv2D-1_layer_call_and_return_conditional_losses_128440k7?4
-?*
(?%
inputs?????????'

? "-?*
#? 
0?????????&
? ?
)__inference_Conv2D-1_layer_call_fn_128433^7?4
-?*
(?%
inputs?????????'

? " ??????????&?
D__inference_Conv2D-2_layer_call_and_return_conditional_losses_128588k$7?4
-?*
(?%
inputs?????????&
? "-?*
#? 
0?????????&
? ?
)__inference_Conv2D-2_layer_call_fn_128581^$7?4
-?*
(?%
inputs?????????&
? " ??????????&?
D__inference_Conv2D-3_layer_call_and_return_conditional_losses_128736k67?4
-?*
(?%
inputs?????????&
? "-?*
#? 
0?????????&
? ?
)__inference_Conv2D-3_layer_call_fn_128729^67?4
-?*
(?%
inputs?????????&
? " ??????????&?
V__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_128886?R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????????????
? ?
V__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_128892`7?4
-?*
(?%
inputs?????????&
? "%?"
?
0?????????&
? ?
;__inference_GlobalAveragePooling-Layer_layer_call_fn_128875wR?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????????????
;__inference_GlobalAveragePooling-Layer_layer_call_fn_128880S7?4
-?*
(?%
inputs?????????&
? "??????????&?
H__inference_Output-Layer_layer_call_and_return_conditional_losses_128911\LM/?,
%?"
 ?
inputs?????????&
? "%?"
?
0?????????
? ?
-__inference_Output-Layer_layer_call_fn_128901OLM/?,
%?"
 ?
inputs?????????&
? "???????????
!__inference__wrapped_model_127308?$*+,-6<=>?LM??<
5?2
0?-
Conv2D-1_input?????????'

? ";?8
6
Output-Layer&?#
Output-Layer??????????
D__inference_re_lu_48_layer_call_and_return_conditional_losses_128574h7?4
-?*
(?%
inputs?????????&
? "-?*
#? 
0?????????&
? ?
)__inference_re_lu_48_layer_call_fn_128569[7?4
-?*
(?%
inputs?????????&
? " ??????????&?
D__inference_re_lu_49_layer_call_and_return_conditional_losses_128722h7?4
-?*
(?%
inputs?????????&
? "-?*
#? 
0?????????&
? ?
)__inference_re_lu_49_layer_call_fn_128717[7?4
-?*
(?%
inputs?????????&
? " ??????????&?
D__inference_re_lu_50_layer_call_and_return_conditional_losses_128870h7?4
-?*
(?%
inputs?????????&
? "-?*
#? 
0?????????&
? ?
)__inference_re_lu_50_layer_call_fn_128865[7?4
-?*
(?%
inputs?????????&
? " ??????????&?
I__inference_sequential_25_layer_call_and_return_conditional_losses_128120?$*+,-6<=>?LMG?D
=?:
0?-
Conv2D-1_input?????????'

p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_25_layer_call_and_return_conditional_losses_128169?$*+,-6<=>?LMG?D
=?:
0?-
Conv2D-1_input?????????'

p

 
? "%?"
?
0?????????
? ?
I__inference_sequential_25_layer_call_and_return_conditional_losses_128360{$*+,-6<=>?LM??<
5?2
(?%
inputs?????????'

p 

 
? "%?"
?
0?????????
? ?
I__inference_sequential_25_layer_call_and_return_conditional_losses_128426{$*+,-6<=>?LM??<
5?2
(?%
inputs?????????'

p

 
? "%?"
?
0?????????
? ?
.__inference_sequential_25_layer_call_fn_127716v$*+,-6<=>?LMG?D
=?:
0?-
Conv2D-1_input?????????'

p 

 
? "???????????
.__inference_sequential_25_layer_call_fn_128071v$*+,-6<=>?LMG?D
=?:
0?-
Conv2D-1_input?????????'

p

 
? "???????????
.__inference_sequential_25_layer_call_fn_128255n$*+,-6<=>?LM??<
5?2
(?%
inputs?????????'

p 

 
? "???????????
.__inference_sequential_25_layer_call_fn_128294n$*+,-6<=>?LM??<
5?2
(?%
inputs?????????'

p

 
? "???????????
$__inference_signature_wrapper_128216?$*+,-6<=>?LMQ?N
? 
G?D
B
Conv2D-1_input0?-
Conv2D-1_input?????????'
";?8
6
Output-Layer&?#
Output-Layer?????????