τ΅#
Ρ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
Ύ
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ΤΟ

!quantize_layer/quantize_layer_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quantize_layer/quantize_layer_min

5quantize_layer/quantize_layer_min/Read/ReadVariableOpReadVariableOp!quantize_layer/quantize_layer_min*
_output_shapes
: *
dtype0

!quantize_layer/quantize_layer_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quantize_layer/quantize_layer_max

5quantize_layer/quantize_layer_max/Read/ReadVariableOpReadVariableOp!quantize_layer/quantize_layer_max*
_output_shapes
: *
dtype0

quantize_layer/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequantize_layer/optimizer_step

1quantize_layer/optimizer_step/Read/ReadVariableOpReadVariableOpquantize_layer/optimizer_step*
_output_shapes
: *
dtype0

quant_Conv2D-1/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_Conv2D-1/optimizer_step

1quant_Conv2D-1/optimizer_step/Read/ReadVariableOpReadVariableOpquant_Conv2D-1/optimizer_step*
_output_shapes
: *
dtype0

quant_Conv2D-1/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:&**
shared_namequant_Conv2D-1/kernel_min

-quant_Conv2D-1/kernel_min/Read/ReadVariableOpReadVariableOpquant_Conv2D-1/kernel_min*
_output_shapes
:&*
dtype0

quant_Conv2D-1/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:&**
shared_namequant_Conv2D-1/kernel_max

-quant_Conv2D-1/kernel_max/Read/ReadVariableOpReadVariableOpquant_Conv2D-1/kernel_max*
_output_shapes
:&*
dtype0

 quant_Btch_Norm-1/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_Btch_Norm-1/optimizer_step

4quant_Btch_Norm-1/optimizer_step/Read/ReadVariableOpReadVariableOp quant_Btch_Norm-1/optimizer_step*
_output_shapes
: *
dtype0

quant_re_lu_6/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_re_lu_6/optimizer_step

0quant_re_lu_6/optimizer_step/Read/ReadVariableOpReadVariableOpquant_re_lu_6/optimizer_step*
_output_shapes
: *
dtype0

quant_re_lu_6/output_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_re_lu_6/output_min
}
,quant_re_lu_6/output_min/Read/ReadVariableOpReadVariableOpquant_re_lu_6/output_min*
_output_shapes
: *
dtype0

quant_re_lu_6/output_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_re_lu_6/output_max
}
,quant_re_lu_6/output_max/Read/ReadVariableOpReadVariableOpquant_re_lu_6/output_max*
_output_shapes
: *
dtype0

quant_Conv2D-2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_Conv2D-2/optimizer_step

1quant_Conv2D-2/optimizer_step/Read/ReadVariableOpReadVariableOpquant_Conv2D-2/optimizer_step*
_output_shapes
: *
dtype0

quant_Conv2D-2/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:&**
shared_namequant_Conv2D-2/kernel_min

-quant_Conv2D-2/kernel_min/Read/ReadVariableOpReadVariableOpquant_Conv2D-2/kernel_min*
_output_shapes
:&*
dtype0

quant_Conv2D-2/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:&**
shared_namequant_Conv2D-2/kernel_max

-quant_Conv2D-2/kernel_max/Read/ReadVariableOpReadVariableOpquant_Conv2D-2/kernel_max*
_output_shapes
:&*
dtype0

 quant_Btch_Norm-2/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_Btch_Norm-2/optimizer_step

4quant_Btch_Norm-2/optimizer_step/Read/ReadVariableOpReadVariableOp quant_Btch_Norm-2/optimizer_step*
_output_shapes
: *
dtype0

quant_re_lu_7/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_re_lu_7/optimizer_step

0quant_re_lu_7/optimizer_step/Read/ReadVariableOpReadVariableOpquant_re_lu_7/optimizer_step*
_output_shapes
: *
dtype0

quant_re_lu_7/output_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_re_lu_7/output_min
}
,quant_re_lu_7/output_min/Read/ReadVariableOpReadVariableOpquant_re_lu_7/output_min*
_output_shapes
: *
dtype0

quant_re_lu_7/output_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_re_lu_7/output_max
}
,quant_re_lu_7/output_max/Read/ReadVariableOpReadVariableOpquant_re_lu_7/output_max*
_output_shapes
: *
dtype0

quant_Conv2D-3/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_Conv2D-3/optimizer_step

1quant_Conv2D-3/optimizer_step/Read/ReadVariableOpReadVariableOpquant_Conv2D-3/optimizer_step*
_output_shapes
: *
dtype0

quant_Conv2D-3/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:&**
shared_namequant_Conv2D-3/kernel_min

-quant_Conv2D-3/kernel_min/Read/ReadVariableOpReadVariableOpquant_Conv2D-3/kernel_min*
_output_shapes
:&*
dtype0

quant_Conv2D-3/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:&**
shared_namequant_Conv2D-3/kernel_max

-quant_Conv2D-3/kernel_max/Read/ReadVariableOpReadVariableOpquant_Conv2D-3/kernel_max*
_output_shapes
:&*
dtype0

 quant_Btch_Norm-3/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_Btch_Norm-3/optimizer_step

4quant_Btch_Norm-3/optimizer_step/Read/ReadVariableOpReadVariableOp quant_Btch_Norm-3/optimizer_step*
_output_shapes
: *
dtype0

quant_re_lu_8/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_re_lu_8/optimizer_step

0quant_re_lu_8/optimizer_step/Read/ReadVariableOpReadVariableOpquant_re_lu_8/optimizer_step*
_output_shapes
: *
dtype0

quant_re_lu_8/output_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_re_lu_8/output_min
}
,quant_re_lu_8/output_min/Read/ReadVariableOpReadVariableOpquant_re_lu_8/output_min*
_output_shapes
: *
dtype0

quant_re_lu_8/output_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namequant_re_lu_8/output_max
}
,quant_re_lu_8/output_max/Read/ReadVariableOpReadVariableOpquant_re_lu_8/output_max*
_output_shapes
: *
dtype0
²
/quant_GlobalAveragePooling-Layer/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/quant_GlobalAveragePooling-Layer/optimizer_step
«
Cquant_GlobalAveragePooling-Layer/optimizer_step/Read/ReadVariableOpReadVariableOp/quant_GlobalAveragePooling-Layer/optimizer_step*
_output_shapes
: *
dtype0
ͺ
+quant_GlobalAveragePooling-Layer/output_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+quant_GlobalAveragePooling-Layer/output_min
£
?quant_GlobalAveragePooling-Layer/output_min/Read/ReadVariableOpReadVariableOp+quant_GlobalAveragePooling-Layer/output_min*
_output_shapes
: *
dtype0
ͺ
+quant_GlobalAveragePooling-Layer/output_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+quant_GlobalAveragePooling-Layer/output_max
£
?quant_GlobalAveragePooling-Layer/output_max/Read/ReadVariableOpReadVariableOp+quant_GlobalAveragePooling-Layer/output_max*
_output_shapes
: *
dtype0

!quant_Output-Layer/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quant_Output-Layer/optimizer_step

5quant_Output-Layer/optimizer_step/Read/ReadVariableOpReadVariableOp!quant_Output-Layer/optimizer_step*
_output_shapes
: *
dtype0

quant_Output-Layer/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_Output-Layer/kernel_min

1quant_Output-Layer/kernel_min/Read/ReadVariableOpReadVariableOpquant_Output-Layer/kernel_min*
_output_shapes
: *
dtype0

quant_Output-Layer/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequant_Output-Layer/kernel_max

1quant_Output-Layer/kernel_max/Read/ReadVariableOpReadVariableOpquant_Output-Layer/kernel_max*
_output_shapes
: *
dtype0
 
&quant_Output-Layer/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&quant_Output-Layer/post_activation_min

:quant_Output-Layer/post_activation_min/Read/ReadVariableOpReadVariableOp&quant_Output-Layer/post_activation_min*
_output_shapes
: *
dtype0
 
&quant_Output-Layer/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&quant_Output-Layer/post_activation_max

:quant_Output-Layer/post_activation_max/Read/ReadVariableOpReadVariableOp&quant_Output-Layer/post_activation_max*
_output_shapes
: *
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

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

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

Btch_Norm-1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*,
shared_nameBtch_Norm-1/moving_variance

/Btch_Norm-1/moving_variance/Read/ReadVariableOpReadVariableOpBtch_Norm-1/moving_variance*
_output_shapes
:&*
dtype0

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

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

Btch_Norm-2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*,
shared_nameBtch_Norm-2/moving_variance

/Btch_Norm-2/moving_variance/Read/ReadVariableOpReadVariableOpBtch_Norm-2/moving_variance*
_output_shapes
:&*
dtype0

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

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

Btch_Norm-3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*,
shared_nameBtch_Norm-3/moving_variance

/Btch_Norm-3/moving_variance/Read/ReadVariableOpReadVariableOpBtch_Norm-3/moving_variance*
_output_shapes
:&*
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

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

Adam/Conv2D-1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*'
shared_nameAdam/Conv2D-1/kernel/m

*Adam/Conv2D-1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv2D-1/kernel/m*&
_output_shapes
:&*
dtype0

Adam/Btch_Norm-1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*)
shared_nameAdam/Btch_Norm-1/gamma/m

,Adam/Btch_Norm-1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-1/gamma/m*
_output_shapes
:&*
dtype0

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

Adam/Conv2D-2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&&*'
shared_nameAdam/Conv2D-2/kernel/m

*Adam/Conv2D-2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv2D-2/kernel/m*&
_output_shapes
:&&*
dtype0

Adam/Btch_Norm-2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*)
shared_nameAdam/Btch_Norm-2/gamma/m

,Adam/Btch_Norm-2/gamma/m/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-2/gamma/m*
_output_shapes
:&*
dtype0

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

Adam/Conv2D-3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&&*'
shared_nameAdam/Conv2D-3/kernel/m

*Adam/Conv2D-3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Conv2D-3/kernel/m*&
_output_shapes
:&&*
dtype0

Adam/Btch_Norm-3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*)
shared_nameAdam/Btch_Norm-3/gamma/m

,Adam/Btch_Norm-3/gamma/m/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-3/gamma/m*
_output_shapes
:&*
dtype0

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

Adam/Output-Layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Output-Layer/bias/m

,Adam/Output-Layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/Output-Layer/bias/m*
_output_shapes
:*
dtype0

Adam/Output-Layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:&*+
shared_nameAdam/Output-Layer/kernel/m

.Adam/Output-Layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/Output-Layer/kernel/m*
_output_shapes

:&*
dtype0

Adam/Conv2D-1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*'
shared_nameAdam/Conv2D-1/kernel/v

*Adam/Conv2D-1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv2D-1/kernel/v*&
_output_shapes
:&*
dtype0

Adam/Btch_Norm-1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*)
shared_nameAdam/Btch_Norm-1/gamma/v

,Adam/Btch_Norm-1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-1/gamma/v*
_output_shapes
:&*
dtype0

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

Adam/Conv2D-2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&&*'
shared_nameAdam/Conv2D-2/kernel/v

*Adam/Conv2D-2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv2D-2/kernel/v*&
_output_shapes
:&&*
dtype0

Adam/Btch_Norm-2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*)
shared_nameAdam/Btch_Norm-2/gamma/v

,Adam/Btch_Norm-2/gamma/v/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-2/gamma/v*
_output_shapes
:&*
dtype0

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

Adam/Conv2D-3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&&*'
shared_nameAdam/Conv2D-3/kernel/v

*Adam/Conv2D-3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Conv2D-3/kernel/v*&
_output_shapes
:&&*
dtype0

Adam/Btch_Norm-3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:&*)
shared_nameAdam/Btch_Norm-3/gamma/v

,Adam/Btch_Norm-3/gamma/v/Read/ReadVariableOpReadVariableOpAdam/Btch_Norm-3/gamma/v*
_output_shapes
:&*
dtype0

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

Adam/Output-Layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/Output-Layer/bias/v

,Adam/Output-Layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/Output-Layer/bias/v*
_output_shapes
:*
dtype0

Adam/Output-Layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:&*+
shared_nameAdam/Output-Layer/kernel/v

.Adam/Output-Layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/Output-Layer/kernel/v*
_output_shapes

:&*
dtype0

NoOpNoOp
Ί 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*τ
valueιBε Bέ
Ι
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
	layer_with_weights-8
	layer-8

layer_with_weights-9

layer-9
layer_with_weights-10
layer-10
layer_with_weights-11
layer-11
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
ͺ
quantize_layer_min
quantize_layer_max
quantizer_vars
optimizer_step
	variables
regularization_losses
trainable_variables
	keras_api
Φ
	layer
optimizer_step
_weight_vars

kernel_min

kernel_max
 _quantize_activations
!_output_quantizers
"	variables
#regularization_losses
$trainable_variables
%	keras_api
Ά
	&layer
'optimizer_step
(_weight_vars
)_quantize_activations
*_output_quantizers
+	variables
,regularization_losses
-trainable_variables
.	keras_api
ς
	/layer
0optimizer_step
1_weight_vars
2_quantize_activations
3_output_quantizers
4
output_min
5
output_max
6_output_quantizer_vars
7	variables
8regularization_losses
9trainable_variables
:	keras_api
Φ
	;layer
<optimizer_step
=_weight_vars
>
kernel_min
?
kernel_max
@_quantize_activations
A_output_quantizers
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
Ά
	Flayer
Goptimizer_step
H_weight_vars
I_quantize_activations
J_output_quantizers
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
ς
	Olayer
Poptimizer_step
Q_weight_vars
R_quantize_activations
S_output_quantizers
T
output_min
U
output_max
V_output_quantizer_vars
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
Φ
	[layer
\optimizer_step
]_weight_vars
^
kernel_min
_
kernel_max
`_quantize_activations
a_output_quantizers
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
Ά
	flayer
goptimizer_step
h_weight_vars
i_quantize_activations
j_output_quantizers
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
ς
	olayer
poptimizer_step
q_weight_vars
r_quantize_activations
s_output_quantizers
t
output_min
u
output_max
v_output_quantizer_vars
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
ω
	{layer
|optimizer_step
}_weight_vars
~_quantize_activations
_output_quantizers

output_min

output_max
_output_quantizer_vars
	variables
regularization_losses
trainable_variables
	keras_api


layer
optimizer_step
_weight_vars

kernel_min

kernel_max
_quantize_activations
post_activation_min
post_activation_max
_output_quantizers
	variables
regularization_losses
trainable_variables
	keras_api
·
	iter
beta_1
beta_2

decay
learning_rate	mδ	mε	mζ	mη	mθ	 mι	£mκ	€mλ	₯mμ	¨mν	©mξ	vο	vπ	vρ	vς	vσ	 vτ	£vυ	€vφ	₯vχ	¨vψ	©vω

0
1
2
3
4
5
6
7
8
9
10
'11
012
413
514
15
<16
>17
?18
19
 20
‘21
’22
G23
P24
T25
U26
£27
\28
^29
_30
€31
₯32
¦33
§34
g35
p36
t37
u38
|39
40
41
¨42
©43
44
45
46
47
48
 
Y
0
1
2
3
4
 5
£6
€7
₯8
¨9
©10
²
ͺlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
?non_trainable_variables
	variables
regularization_losses
trainable_variables
 
yw
VARIABLE_VALUE!quantize_layer/quantize_layer_minBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!quantize_layer/quantize_layer_maxBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUE

min_var
max_var
qo
VARIABLE_VALUEquantize_layer/optimizer_step>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
 
²
―layers
°metrics
 ±layer_regularization_losses
	variables
²non_trainable_variables
³layer_metrics
regularization_losses
trainable_variables
c
kernel
΄	variables
΅regularization_losses
Άtrainable_variables
·	keras_api
qo
VARIABLE_VALUEquant_Conv2D-1/optimizer_step>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

Έ0
ig
VARIABLE_VALUEquant_Conv2D-1/kernel_min:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_Conv2D-1/kernel_max:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
2
3
 

0
²
Ήlayers
Ίmetrics
 »layer_regularization_losses
"	variables
Όnon_trainable_variables
½layer_metrics
#regularization_losses
$trainable_variables
 
	Ύaxis

gamma
	beta
moving_mean
moving_variance
Ώ	variables
ΐregularization_losses
Αtrainable_variables
Β	keras_api
tr
VARIABLE_VALUE quant_Btch_Norm-1/optimizer_step>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
'
0
1
2
3
'4
 

0
1
²
Γlayers
Δmetrics
 Εlayer_regularization_losses
+	variables
Ζnon_trainable_variables
Ηlayer_metrics
,regularization_losses
-trainable_variables
V
Θ	variables
Ιregularization_losses
Κtrainable_variables
Λ	keras_api
pn
VARIABLE_VALUEquant_re_lu_6/optimizer_step>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
hf
VARIABLE_VALUEquant_re_lu_6/output_min:layer_with_weights-3/output_min/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEquant_re_lu_6/output_max:layer_with_weights-3/output_max/.ATTRIBUTES/VARIABLE_VALUE

4min_var
5max_var

00
41
52
 
 
²
Μlayers
Νmetrics
 Ξlayer_regularization_losses
7	variables
Οnon_trainable_variables
Πlayer_metrics
8regularization_losses
9trainable_variables
c
kernel
Ρ	variables
?regularization_losses
Σtrainable_variables
Τ	keras_api
qo
VARIABLE_VALUEquant_Conv2D-2/optimizer_step>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

Υ0
ig
VARIABLE_VALUEquant_Conv2D-2/kernel_min:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_Conv2D-2/kernel_max:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
<1
>2
?3
 

0
²
Φlayers
Χmetrics
 Ψlayer_regularization_losses
B	variables
Ωnon_trainable_variables
Ϊlayer_metrics
Cregularization_losses
Dtrainable_variables
 
	Ϋaxis

gamma
	 beta
‘moving_mean
’moving_variance
ά	variables
έregularization_losses
ήtrainable_variables
ί	keras_api
tr
VARIABLE_VALUE quant_Btch_Norm-2/optimizer_step>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
'
0
 1
‘2
’3
G4
 

0
 1
²
ΰlayers
αmetrics
 βlayer_regularization_losses
K	variables
γnon_trainable_variables
δlayer_metrics
Lregularization_losses
Mtrainable_variables
V
ε	variables
ζregularization_losses
ηtrainable_variables
θ	keras_api
pn
VARIABLE_VALUEquant_re_lu_7/optimizer_step>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
hf
VARIABLE_VALUEquant_re_lu_7/output_min:layer_with_weights-6/output_min/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEquant_re_lu_7/output_max:layer_with_weights-6/output_max/.ATTRIBUTES/VARIABLE_VALUE

Tmin_var
Umax_var

P0
T1
U2
 
 
²
ιlayers
κmetrics
 λlayer_regularization_losses
W	variables
μnon_trainable_variables
νlayer_metrics
Xregularization_losses
Ytrainable_variables
c
£kernel
ξ	variables
οregularization_losses
πtrainable_variables
ρ	keras_api
qo
VARIABLE_VALUEquant_Conv2D-3/optimizer_step>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

ς0
ig
VARIABLE_VALUEquant_Conv2D-3/kernel_min:layer_with_weights-7/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEquant_Conv2D-3/kernel_max:layer_with_weights-7/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
 

£0
\1
^2
_3
 

£0
²
σlayers
τmetrics
 υlayer_regularization_losses
b	variables
φnon_trainable_variables
χlayer_metrics
cregularization_losses
dtrainable_variables
 
	ψaxis

€gamma
	₯beta
¦moving_mean
§moving_variance
ω	variables
ϊregularization_losses
ϋtrainable_variables
ό	keras_api
tr
VARIABLE_VALUE quant_Btch_Norm-3/optimizer_step>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
'
€0
₯1
¦2
§3
g4
 

€0
₯1
²
ύlayers
ώmetrics
 ?layer_regularization_losses
k	variables
non_trainable_variables
layer_metrics
lregularization_losses
mtrainable_variables
V
	variables
regularization_losses
trainable_variables
	keras_api
pn
VARIABLE_VALUEquant_re_lu_8/optimizer_step>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
hf
VARIABLE_VALUEquant_re_lu_8/output_min:layer_with_weights-9/output_min/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEquant_re_lu_8/output_max:layer_with_weights-9/output_max/.ATTRIBUTES/VARIABLE_VALUE

tmin_var
umax_var

p0
t1
u2
 
 
²
layers
metrics
 layer_regularization_losses
w	variables
non_trainable_variables
layer_metrics
xregularization_losses
ytrainable_variables
V
	variables
regularization_losses
trainable_variables
	keras_api

VARIABLE_VALUE/quant_GlobalAveragePooling-Layer/optimizer_step?layer_with_weights-10/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
|z
VARIABLE_VALUE+quant_GlobalAveragePooling-Layer/output_min;layer_with_weights-10/output_min/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE+quant_GlobalAveragePooling-Layer/output_max;layer_with_weights-10/output_max/.ATTRIBUTES/VARIABLE_VALUE

min_var
max_var

|0
1
2
 
 
΅
layers
metrics
 layer_regularization_losses
	variables
non_trainable_variables
layer_metrics
regularization_losses
trainable_variables
n
©kernel
	¨bias
	variables
regularization_losses
trainable_variables
	keras_api
vt
VARIABLE_VALUE!quant_Output-Layer/optimizer_step?layer_with_weights-11/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

0
nl
VARIABLE_VALUEquant_Output-Layer/kernel_min;layer_with_weights-11/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEquant_Output-Layer/kernel_max;layer_with_weights-11/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
~
VARIABLE_VALUE&quant_Output-Layer/post_activation_minDlayer_with_weights-11/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUE&quant_Output-Layer/post_activation_maxDlayer_with_weights-11/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
8
¨0
©1
2
3
4
5
6
 

¨0
©1
΅
layers
metrics
 layer_regularization_losses
	variables
non_trainable_variables
layer_metrics
regularization_losses
trainable_variables
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
KI
VARIABLE_VALUEConv2D-1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEBtch_Norm-1/gamma&variables/7/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEBtch_Norm-1/beta&variables/8/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEBtch_Norm-1/moving_mean&variables/9/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEBtch_Norm-1/moving_variance'variables/10/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEConv2D-2/kernel'variables/15/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEBtch_Norm-2/gamma'variables/19/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEBtch_Norm-2/beta'variables/20/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEBtch_Norm-2/moving_mean'variables/21/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEBtch_Norm-2/moving_variance'variables/22/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEConv2D-3/kernel'variables/27/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEBtch_Norm-3/gamma'variables/31/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEBtch_Norm-3/beta'variables/32/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEBtch_Norm-3/moving_mean'variables/33/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEBtch_Norm-3/moving_variance'variables/34/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEOutput-Layer/bias'variables/42/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEOutput-Layer/kernel'variables/43/.ATTRIBUTES/VARIABLE_VALUE
V
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
11

0
1
 
 
³
0
1
2
3
4
5
6
7
'8
09
410
511
<12
>13
?14
‘15
’16
G17
P18
T19
U20
\21
^22
_23
¦24
§25
g26
p27
t28
u29
|30
31
32
33
34
35
36
37
 
 
 

0
1
2
 
 
 
 
΅
 layers
‘metrics
 ’layer_regularization_losses
΄	variables
£non_trainable_variables
€layer_metrics
΅regularization_losses
Άtrainable_variables

0
₯2

0
 
 

0
1
2
 
 
 
0
1
2
3
 

0
1
΅
¦layers
§metrics
 ¨layer_regularization_losses
Ώ	variables
©non_trainable_variables
ͺlayer_metrics
ΐregularization_losses
Αtrainable_variables

&0
 
 

0
1
'2
 
 
 
 
΅
«layers
¬metrics
 ­layer_regularization_losses
Θ	variables
?non_trainable_variables
―layer_metrics
Ιregularization_losses
Κtrainable_variables

/0
 
 

00
41
52
 
 
 
 
΅
°layers
±metrics
 ²layer_regularization_losses
Ρ	variables
³non_trainable_variables
΄layer_metrics
?regularization_losses
Σtrainable_variables

0
΅2

;0
 
 

<0
>1
?2
 
 
 
0
 1
‘2
’3
 

0
 1
΅
Άlayers
·metrics
 Έlayer_regularization_losses
ά	variables
Ήnon_trainable_variables
Ίlayer_metrics
έregularization_losses
ήtrainable_variables

F0
 
 

‘0
’1
G2
 
 
 
 
΅
»layers
Όmetrics
 ½layer_regularization_losses
ε	variables
Ύnon_trainable_variables
Ώlayer_metrics
ζregularization_losses
ηtrainable_variables

O0
 
 

P0
T1
U2
 
 
 
 
΅
ΐlayers
Αmetrics
 Βlayer_regularization_losses
ξ	variables
Γnon_trainable_variables
Δlayer_metrics
οregularization_losses
πtrainable_variables

£0
Ε2

[0
 
 

\0
^1
_2
 
 
 
€0
₯1
¦2
§3
 

€0
₯1
΅
Ζlayers
Ηmetrics
 Θlayer_regularization_losses
ω	variables
Ιnon_trainable_variables
Κlayer_metrics
ϊregularization_losses
ϋtrainable_variables

f0
 
 

¦0
§1
g2
 
 
 
 
΅
Λlayers
Μmetrics
 Νlayer_regularization_losses
	variables
Ξnon_trainable_variables
Οlayer_metrics
regularization_losses
trainable_variables

o0
 
 

p0
t1
u2
 
 
 
 
΅
Πlayers
Ρmetrics
 ?layer_regularization_losses
	variables
Σnon_trainable_variables
Τlayer_metrics
regularization_losses
trainable_variables

{0
 
 

|0
1
2
 

¨0
 

¨0
΅
Υlayers
Φmetrics
 Χlayer_regularization_losses
	variables
Ψnon_trainable_variables
Ωlayer_metrics
regularization_losses
trainable_variables

©0
Ϊ2

0
 
 
(
0
1
2
3
4
 
8

Ϋtotal

άcount
έ	variables
ή	keras_api
I

ίtotal

ΰcount
α
_fn_kwargs
β	variables
γ	keras_api
 
 
 
 
 

min_var
max_var
 
 
 

0
1
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

>min_var
?max_var
 
 
 

‘0
’1
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

^min_var
_max_var
 
 
 

¦0
§1
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

min_var
max_var
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Ϋ0
ά1

έ	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

ί0
ΰ1

β	variables
nl
VARIABLE_VALUEAdam/Conv2D-1/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/Btch_Norm-1/gamma/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Btch_Norm-1/beta/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Conv2D-2/kernel/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/Btch_Norm-2/gamma/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/Btch_Norm-2/beta/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Conv2D-3/kernel/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/Btch_Norm-3/gamma/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/Btch_Norm-3/beta/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/Output-Layer/bias/mCvariables/42/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/Output-Layer/kernel/mCvariables/43/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/Conv2D-1/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/Btch_Norm-1/gamma/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Btch_Norm-1/beta/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Conv2D-2/kernel/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/Btch_Norm-2/gamma/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/Btch_Norm-2/beta/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/Conv2D-3/kernel/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/Btch_Norm-3/gamma/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/Btch_Norm-3/beta/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/Output-Layer/bias/vCvariables/42/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/Output-Layer/kernel/vCvariables/43/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_Conv2D-1_inputPlaceholder*/
_output_shapes
:?????????1
*
dtype0*$
shape:?????????1

Ϊ

StatefulPartitionedCallStatefulPartitionedCallserving_default_Conv2D-1_input!quantize_layer/quantize_layer_min!quantize_layer/quantize_layer_maxConv2D-1/kernelquant_Conv2D-1/kernel_minquant_Conv2D-1/kernel_maxBtch_Norm-1/gammaBtch_Norm-1/betaBtch_Norm-1/moving_meanBtch_Norm-1/moving_variancequant_re_lu_6/output_minquant_re_lu_6/output_maxConv2D-2/kernelquant_Conv2D-2/kernel_minquant_Conv2D-2/kernel_maxBtch_Norm-2/gammaBtch_Norm-2/betaBtch_Norm-2/moving_meanBtch_Norm-2/moving_variancequant_re_lu_7/output_minquant_re_lu_7/output_maxConv2D-3/kernelquant_Conv2D-3/kernel_minquant_Conv2D-3/kernel_maxBtch_Norm-3/gammaBtch_Norm-3/betaBtch_Norm-3/moving_meanBtch_Norm-3/moving_variancequant_re_lu_8/output_minquant_re_lu_8/output_max+quant_GlobalAveragePooling-Layer/output_min+quant_GlobalAveragePooling-Layer/output_maxOutput-Layer/kernelquant_Output-Layer/kernel_minquant_Output-Layer/kernel_maxOutput-Layer/bias&quant_Output-Layer/post_activation_min&quant_Output-Layer/post_activation_max*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*G
_read_only_resource_inputs)
'%	
 !"#$%*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_58768
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
₯
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename5quantize_layer/quantize_layer_min/Read/ReadVariableOp5quantize_layer/quantize_layer_max/Read/ReadVariableOp1quantize_layer/optimizer_step/Read/ReadVariableOp1quant_Conv2D-1/optimizer_step/Read/ReadVariableOp-quant_Conv2D-1/kernel_min/Read/ReadVariableOp-quant_Conv2D-1/kernel_max/Read/ReadVariableOp4quant_Btch_Norm-1/optimizer_step/Read/ReadVariableOp0quant_re_lu_6/optimizer_step/Read/ReadVariableOp,quant_re_lu_6/output_min/Read/ReadVariableOp,quant_re_lu_6/output_max/Read/ReadVariableOp1quant_Conv2D-2/optimizer_step/Read/ReadVariableOp-quant_Conv2D-2/kernel_min/Read/ReadVariableOp-quant_Conv2D-2/kernel_max/Read/ReadVariableOp4quant_Btch_Norm-2/optimizer_step/Read/ReadVariableOp0quant_re_lu_7/optimizer_step/Read/ReadVariableOp,quant_re_lu_7/output_min/Read/ReadVariableOp,quant_re_lu_7/output_max/Read/ReadVariableOp1quant_Conv2D-3/optimizer_step/Read/ReadVariableOp-quant_Conv2D-3/kernel_min/Read/ReadVariableOp-quant_Conv2D-3/kernel_max/Read/ReadVariableOp4quant_Btch_Norm-3/optimizer_step/Read/ReadVariableOp0quant_re_lu_8/optimizer_step/Read/ReadVariableOp,quant_re_lu_8/output_min/Read/ReadVariableOp,quant_re_lu_8/output_max/Read/ReadVariableOpCquant_GlobalAveragePooling-Layer/optimizer_step/Read/ReadVariableOp?quant_GlobalAveragePooling-Layer/output_min/Read/ReadVariableOp?quant_GlobalAveragePooling-Layer/output_max/Read/ReadVariableOp5quant_Output-Layer/optimizer_step/Read/ReadVariableOp1quant_Output-Layer/kernel_min/Read/ReadVariableOp1quant_Output-Layer/kernel_max/Read/ReadVariableOp:quant_Output-Layer/post_activation_min/Read/ReadVariableOp:quant_Output-Layer/post_activation_max/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#Conv2D-1/kernel/Read/ReadVariableOp%Btch_Norm-1/gamma/Read/ReadVariableOp$Btch_Norm-1/beta/Read/ReadVariableOp+Btch_Norm-1/moving_mean/Read/ReadVariableOp/Btch_Norm-1/moving_variance/Read/ReadVariableOp#Conv2D-2/kernel/Read/ReadVariableOp%Btch_Norm-2/gamma/Read/ReadVariableOp$Btch_Norm-2/beta/Read/ReadVariableOp+Btch_Norm-2/moving_mean/Read/ReadVariableOp/Btch_Norm-2/moving_variance/Read/ReadVariableOp#Conv2D-3/kernel/Read/ReadVariableOp%Btch_Norm-3/gamma/Read/ReadVariableOp$Btch_Norm-3/beta/Read/ReadVariableOp+Btch_Norm-3/moving_mean/Read/ReadVariableOp/Btch_Norm-3/moving_variance/Read/ReadVariableOp%Output-Layer/bias/Read/ReadVariableOp'Output-Layer/kernel/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/Conv2D-1/kernel/m/Read/ReadVariableOp,Adam/Btch_Norm-1/gamma/m/Read/ReadVariableOp+Adam/Btch_Norm-1/beta/m/Read/ReadVariableOp*Adam/Conv2D-2/kernel/m/Read/ReadVariableOp,Adam/Btch_Norm-2/gamma/m/Read/ReadVariableOp+Adam/Btch_Norm-2/beta/m/Read/ReadVariableOp*Adam/Conv2D-3/kernel/m/Read/ReadVariableOp,Adam/Btch_Norm-3/gamma/m/Read/ReadVariableOp+Adam/Btch_Norm-3/beta/m/Read/ReadVariableOp,Adam/Output-Layer/bias/m/Read/ReadVariableOp.Adam/Output-Layer/kernel/m/Read/ReadVariableOp*Adam/Conv2D-1/kernel/v/Read/ReadVariableOp,Adam/Btch_Norm-1/gamma/v/Read/ReadVariableOp+Adam/Btch_Norm-1/beta/v/Read/ReadVariableOp*Adam/Conv2D-2/kernel/v/Read/ReadVariableOp,Adam/Btch_Norm-2/gamma/v/Read/ReadVariableOp+Adam/Btch_Norm-2/beta/v/Read/ReadVariableOp*Adam/Conv2D-3/kernel/v/Read/ReadVariableOp,Adam/Btch_Norm-3/gamma/v/Read/ReadVariableOp+Adam/Btch_Norm-3/beta/v/Read/ReadVariableOp,Adam/Output-Layer/bias/v/Read/ReadVariableOp.Adam/Output-Layer/kernel/v/Read/ReadVariableOpConst*]
TinV
T2R	*
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
GPU 2J 8 *'
f"R 
__inference__traced_save_60509
ΰ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!quantize_layer/quantize_layer_min!quantize_layer/quantize_layer_maxquantize_layer/optimizer_stepquant_Conv2D-1/optimizer_stepquant_Conv2D-1/kernel_minquant_Conv2D-1/kernel_max quant_Btch_Norm-1/optimizer_stepquant_re_lu_6/optimizer_stepquant_re_lu_6/output_minquant_re_lu_6/output_maxquant_Conv2D-2/optimizer_stepquant_Conv2D-2/kernel_minquant_Conv2D-2/kernel_max quant_Btch_Norm-2/optimizer_stepquant_re_lu_7/optimizer_stepquant_re_lu_7/output_minquant_re_lu_7/output_maxquant_Conv2D-3/optimizer_stepquant_Conv2D-3/kernel_minquant_Conv2D-3/kernel_max quant_Btch_Norm-3/optimizer_stepquant_re_lu_8/optimizer_stepquant_re_lu_8/output_minquant_re_lu_8/output_max/quant_GlobalAveragePooling-Layer/optimizer_step+quant_GlobalAveragePooling-Layer/output_min+quant_GlobalAveragePooling-Layer/output_max!quant_Output-Layer/optimizer_stepquant_Output-Layer/kernel_minquant_Output-Layer/kernel_max&quant_Output-Layer/post_activation_min&quant_Output-Layer/post_activation_max	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateConv2D-1/kernelBtch_Norm-1/gammaBtch_Norm-1/betaBtch_Norm-1/moving_meanBtch_Norm-1/moving_varianceConv2D-2/kernelBtch_Norm-2/gammaBtch_Norm-2/betaBtch_Norm-2/moving_meanBtch_Norm-2/moving_varianceConv2D-3/kernelBtch_Norm-3/gammaBtch_Norm-3/betaBtch_Norm-3/moving_meanBtch_Norm-3/moving_varianceOutput-Layer/biasOutput-Layer/kerneltotalcounttotal_1count_1Adam/Conv2D-1/kernel/mAdam/Btch_Norm-1/gamma/mAdam/Btch_Norm-1/beta/mAdam/Conv2D-2/kernel/mAdam/Btch_Norm-2/gamma/mAdam/Btch_Norm-2/beta/mAdam/Conv2D-3/kernel/mAdam/Btch_Norm-3/gamma/mAdam/Btch_Norm-3/beta/mAdam/Output-Layer/bias/mAdam/Output-Layer/kernel/mAdam/Conv2D-1/kernel/vAdam/Btch_Norm-1/gamma/vAdam/Btch_Norm-1/beta/vAdam/Conv2D-2/kernel/vAdam/Btch_Norm-2/gamma/vAdam/Btch_Norm-2/beta/vAdam/Conv2D-3/kernel/vAdam/Btch_Norm-3/gamma/vAdam/Btch_Norm-3/beta/vAdam/Output-Layer/bias/vAdam/Output-Layer/kernel/v*\
TinU
S2Q*
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_60759οΘ
Λ

L__inference_quant_Btch_Norm-1_layer_call_and_return_conditional_losses_59456

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&:::::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
οM

G__inference_sequential_4_layer_call_and_return_conditional_losses_58340
conv2d_1_input
quantize_layer_58253
quantize_layer_58255
quant_conv2d_1_58258
quant_conv2d_1_58260
quant_conv2d_1_58262
quant_btch_norm_1_58265
quant_btch_norm_1_58267
quant_btch_norm_1_58269
quant_btch_norm_1_58271
quant_re_lu_6_58274
quant_re_lu_6_58276
quant_conv2d_2_58279
quant_conv2d_2_58281
quant_conv2d_2_58283
quant_btch_norm_2_58286
quant_btch_norm_2_58288
quant_btch_norm_2_58290
quant_btch_norm_2_58292
quant_re_lu_7_58295
quant_re_lu_7_58297
quant_conv2d_3_58300
quant_conv2d_3_58302
quant_conv2d_3_58304
quant_btch_norm_3_58307
quant_btch_norm_3_58309
quant_btch_norm_3_58311
quant_btch_norm_3_58313
quant_re_lu_8_58316
quant_re_lu_8_58318*
&quant_globalaveragepooling_layer_58321*
&quant_globalaveragepooling_layer_58323
quant_output_layer_58326
quant_output_layer_58328
quant_output_layer_58330
quant_output_layer_58332
quant_output_layer_58334
quant_output_layer_58336
identity’)quant_Btch_Norm-1/StatefulPartitionedCall’)quant_Btch_Norm-2/StatefulPartitionedCall’)quant_Btch_Norm-3/StatefulPartitionedCall’&quant_Conv2D-1/StatefulPartitionedCall’&quant_Conv2D-2/StatefulPartitionedCall’&quant_Conv2D-3/StatefulPartitionedCall’8quant_GlobalAveragePooling-Layer/StatefulPartitionedCall’*quant_Output-Layer/StatefulPartitionedCall’%quant_re_lu_6/StatefulPartitionedCall’%quant_re_lu_7/StatefulPartitionedCall’%quant_re_lu_8/StatefulPartitionedCall’&quantize_layer/StatefulPartitionedCallΏ
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputquantize_layer_58253quantize_layer_58255*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quantize_layer_layer_call_and_return_conditional_losses_574272(
&quantize_layer/StatefulPartitionedCallψ
&quant_Conv2D-1/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_1_58258quant_conv2d_1_58260quant_conv2d_1_58262*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-1_layer_call_and_return_conditional_losses_574882(
&quant_Conv2D-1/StatefulPartitionedCall₯
)quant_Btch_Norm-1/StatefulPartitionedCallStatefulPartitionedCall/quant_Conv2D-1/StatefulPartitionedCall:output:0quant_btch_norm_1_58265quant_btch_norm_1_58267quant_btch_norm_1_58269quant_btch_norm_1_58271*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-1_layer_call_and_return_conditional_losses_575572+
)quant_Btch_Norm-1/StatefulPartitionedCallή
%quant_re_lu_6/StatefulPartitionedCallStatefulPartitionedCall2quant_Btch_Norm-1/StatefulPartitionedCall:output:0quant_re_lu_6_58274quant_re_lu_6_58276*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_6_layer_call_and_return_conditional_losses_576302'
%quant_re_lu_6/StatefulPartitionedCallχ
&quant_Conv2D-2/StatefulPartitionedCallStatefulPartitionedCall.quant_re_lu_6/StatefulPartitionedCall:output:0quant_conv2d_2_58279quant_conv2d_2_58281quant_conv2d_2_58283*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-2_layer_call_and_return_conditional_losses_576912(
&quant_Conv2D-2/StatefulPartitionedCall₯
)quant_Btch_Norm-2/StatefulPartitionedCallStatefulPartitionedCall/quant_Conv2D-2/StatefulPartitionedCall:output:0quant_btch_norm_2_58286quant_btch_norm_2_58288quant_btch_norm_2_58290quant_btch_norm_2_58292*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-2_layer_call_and_return_conditional_losses_577602+
)quant_Btch_Norm-2/StatefulPartitionedCallή
%quant_re_lu_7/StatefulPartitionedCallStatefulPartitionedCall2quant_Btch_Norm-2/StatefulPartitionedCall:output:0quant_re_lu_7_58295quant_re_lu_7_58297*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_7_layer_call_and_return_conditional_losses_578332'
%quant_re_lu_7/StatefulPartitionedCallχ
&quant_Conv2D-3/StatefulPartitionedCallStatefulPartitionedCall.quant_re_lu_7/StatefulPartitionedCall:output:0quant_conv2d_3_58300quant_conv2d_3_58302quant_conv2d_3_58304*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-3_layer_call_and_return_conditional_losses_578942(
&quant_Conv2D-3/StatefulPartitionedCall₯
)quant_Btch_Norm-3/StatefulPartitionedCallStatefulPartitionedCall/quant_Conv2D-3/StatefulPartitionedCall:output:0quant_btch_norm_3_58307quant_btch_norm_3_58309quant_btch_norm_3_58311quant_btch_norm_3_58313*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-3_layer_call_and_return_conditional_losses_579632+
)quant_Btch_Norm-3/StatefulPartitionedCallή
%quant_re_lu_8/StatefulPartitionedCallStatefulPartitionedCall2quant_Btch_Norm-3/StatefulPartitionedCall:output:0quant_re_lu_8_58316quant_re_lu_8_58318*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_8_layer_call_and_return_conditional_losses_580362'
%quant_re_lu_8/StatefulPartitionedCall±
8quant_GlobalAveragePooling-Layer/StatefulPartitionedCallStatefulPartitionedCall.quant_re_lu_8/StatefulPartitionedCall:output:0&quant_globalaveragepooling_layer_58321&quant_globalaveragepooling_layer_58323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *d
f_R]
[__inference_quant_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_580992:
8quant_GlobalAveragePooling-Layer/StatefulPartitionedCallξ
*quant_Output-Layer/StatefulPartitionedCallStatefulPartitionedCallAquant_GlobalAveragePooling-Layer/StatefulPartitionedCall:output:0quant_output_layer_58326quant_output_layer_58328quant_output_layer_58330quant_output_layer_58332quant_output_layer_58334quant_output_layer_58336*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_quant_Output-Layer_layer_call_and_return_conditional_losses_582002,
*quant_Output-Layer/StatefulPartitionedCall
IdentityIdentity3quant_Output-Layer/StatefulPartitionedCall:output:0*^quant_Btch_Norm-1/StatefulPartitionedCall*^quant_Btch_Norm-2/StatefulPartitionedCall*^quant_Btch_Norm-3/StatefulPartitionedCall'^quant_Conv2D-1/StatefulPartitionedCall'^quant_Conv2D-2/StatefulPartitionedCall'^quant_Conv2D-3/StatefulPartitionedCall9^quant_GlobalAveragePooling-Layer/StatefulPartitionedCall+^quant_Output-Layer/StatefulPartitionedCall&^quant_re_lu_6/StatefulPartitionedCall&^quant_re_lu_7/StatefulPartitionedCall&^quant_re_lu_8/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Δ
_input_shapes²
―:?????????1
:::::::::::::::::::::::::::::::::::::2V
)quant_Btch_Norm-1/StatefulPartitionedCall)quant_Btch_Norm-1/StatefulPartitionedCall2V
)quant_Btch_Norm-2/StatefulPartitionedCall)quant_Btch_Norm-2/StatefulPartitionedCall2V
)quant_Btch_Norm-3/StatefulPartitionedCall)quant_Btch_Norm-3/StatefulPartitionedCall2P
&quant_Conv2D-1/StatefulPartitionedCall&quant_Conv2D-1/StatefulPartitionedCall2P
&quant_Conv2D-2/StatefulPartitionedCall&quant_Conv2D-2/StatefulPartitionedCall2P
&quant_Conv2D-3/StatefulPartitionedCall&quant_Conv2D-3/StatefulPartitionedCall2t
8quant_GlobalAveragePooling-Layer/StatefulPartitionedCall8quant_GlobalAveragePooling-Layer/StatefulPartitionedCall2X
*quant_Output-Layer/StatefulPartitionedCall*quant_Output-Layer/StatefulPartitionedCall2N
%quant_re_lu_6/StatefulPartitionedCall%quant_re_lu_6/StatefulPartitionedCall2N
%quant_re_lu_7/StatefulPartitionedCall%quant_re_lu_7/StatefulPartitionedCall2N
%quant_re_lu_8/StatefulPartitionedCall%quant_re_lu_8/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????1

(
_user_specified_nameConv2D-1_input

?
F__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_57265

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&:::::i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
»
q
U__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_57387

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

?
F__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_60092

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&:::::i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs

Φ
I__inference_quant_Conv2D-3_layer_call_and_return_conditional_losses_57894

inputsL
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resourceN
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resourceN
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource
identity
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:&&*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2€
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&&*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelΑ
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
Conv2Dk
IdentityIdentityConv2D:output:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????&::::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
ό
©
L__inference_quant_Btch_Norm-2_layer_call_and_return_conditional_losses_57742

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ψ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
ζ#
±
I__inference_quant_Conv2D-3_layer_call_and_return_conditional_losses_59736

inputs3
/lastvaluequant_batchmin_readvariableop_resource)
%lastvaluequant_assignminlast_resource)
%lastvaluequant_assignmaxlast_resource
identity’LastValueQuant/AssignMaxLast’LastValueQuant/AssignMinLastΘ
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOp«
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesΒ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:&2
LastValueQuant/BatchMinΘ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOp«
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesΒ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:&2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2
LastValueQuant/truediv/y₯
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:&2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:&2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:&2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:&2
LastValueQuant/MaximumΆ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastΆ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastϊ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2€
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&&*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelΑ
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
Conv2D©
IdentityIdentityConv2D:output:0^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????&:::2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs


-__inference_quant_re_lu_7_layer_call_fn_59703

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallό
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_7_layer_call_and_return_conditional_losses_578232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
»
χ
H__inference_quant_re_lu_7_layer_call_and_return_conditional_losses_59694

inputsE
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resourceG
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????&2
Reluξ
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpτ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ό
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&:::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
«)
Ι
H__inference_quant_re_lu_8_layer_call_and_return_conditional_losses_58026

inputs:
6movingavgquantize_assignminema_readvariableop_resource:
6movingavgquantize_assignmaxema_readvariableop_resource
identity’2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp’2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpV
ReluReluinputs*
T0*/
_output_shapes
:?????????&2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayΝ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subΗ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul²
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayΝ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subΗ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul²
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ό
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsω
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:03^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&::2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
Λ

L__inference_quant_Btch_Norm-3_layer_call_and_return_conditional_losses_57963

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&:::::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
¬*
ά
[__inference_quant_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_58088

inputs:
6movingavgquantize_assignminema_readvariableop_resource:
6movingavgquantize_assignmaxema_readvariableop_resource
identity’2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp’2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????&2
Mean
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinMean:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxMean:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayΝ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subΗ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul²
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayΝ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subΗ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul²
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1―
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMean:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????&2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsρ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:03^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
T0*'
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&::2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs


@__inference_quant_GlobalAveragePooling-Layer_layer_call_fn_59944

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *d
f_R]
[__inference_quant_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_580992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
§

.__inference_quant_Conv2D-1_layer_call_fn_59418

inputs
unknown
	unknown_0
	unknown_1
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-1_layer_call_and_return_conditional_losses_574882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????1
:::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs
Ν
€
1__inference_quant_Btch_Norm-1_layer_call_fn_59482

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-1_layer_call_and_return_conditional_losses_575572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
Ν
€
1__inference_quant_Btch_Norm-2_layer_call_fn_59658

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-2_layer_call_and_return_conditional_losses_577602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
Φ

[__inference_quant_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_59926

inputsE
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resourceG
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????&2
Meanξ
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpτ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1―
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMean:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????&2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&:::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs

Φ
I__inference_quant_Conv2D-2_layer_call_and_return_conditional_losses_57691

inputsL
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resourceN
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resourceN
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource
identity
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:&&*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2€
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&&*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelΑ
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
Conv2Dk
IdentityIdentityConv2D:output:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????&::::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
§

.__inference_quant_Conv2D-2_layer_call_fn_59594

inputs
unknown
	unknown_0
	unknown_1
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-2_layer_call_and_return_conditional_losses_576912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????&:::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
ό
©
L__inference_quant_Btch_Norm-1_layer_call_and_return_conditional_losses_59438

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ψ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs


-__inference_quant_re_lu_6_layer_call_fn_59527

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallό
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_6_layer_call_and_return_conditional_losses_576202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
Λ

L__inference_quant_Btch_Norm-3_layer_call_and_return_conditional_losses_59808

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&:::::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs


-__inference_quant_re_lu_8_layer_call_fn_59888

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_8_layer_call_and_return_conditional_losses_580362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
λ
Γ
2__inference_quant_Output-Layer_layer_call_fn_60037

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity’StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_quant_Output-Layer_layer_call_and_return_conditional_losses_581802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????&
 
_user_specified_nameinputs


@__inference_quant_GlobalAveragePooling-Layer_layer_call_fn_59935

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
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
GPU 2J 8 *d
f_R]
[__inference_quant_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_580882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs


+__inference_Btch_Norm-1_layer_call_fn_60118

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallͺ
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
GPU 2J 8 *O
fJRH
F__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_571612
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs


-__inference_quant_re_lu_6_layer_call_fn_59536

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_6_layer_call_and_return_conditional_losses_576302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
¬*
ά
[__inference_quant_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_59915

inputs:
6movingavgquantize_assignminema_readvariableop_resource:
6movingavgquantize_assignmaxema_readvariableop_resource
identity’2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp’2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????&2
Mean
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinMean:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxMean:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayΝ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subΗ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul²
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayΝ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subΗ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul²
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1―
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMean:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????&2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsρ
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:03^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
T0*'
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&::2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
§

.__inference_quant_Conv2D-3_layer_call_fn_59770

inputs
unknown
	unknown_0
	unknown_1
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-3_layer_call_and_return_conditional_losses_578942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????&:::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
Ψ

ψ
I__inference_quantize_layer_layer_call_and_return_conditional_losses_57427

inputsE
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resourceG
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource
identityξ
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpτ
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1°
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????1
2+
)AllValuesQuantize/FakeQuantWithMinMaxVars
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*/
_output_shapes
:?????????1
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????1
:::W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs
Ύ
£
F__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_57130

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o:*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
³M

G__inference_sequential_4_layer_call_and_return_conditional_losses_58433

inputs
quantize_layer_58346
quantize_layer_58348
quant_conv2d_1_58351
quant_conv2d_1_58353
quant_conv2d_1_58355
quant_btch_norm_1_58358
quant_btch_norm_1_58360
quant_btch_norm_1_58362
quant_btch_norm_1_58364
quant_re_lu_6_58367
quant_re_lu_6_58369
quant_conv2d_2_58372
quant_conv2d_2_58374
quant_conv2d_2_58376
quant_btch_norm_2_58379
quant_btch_norm_2_58381
quant_btch_norm_2_58383
quant_btch_norm_2_58385
quant_re_lu_7_58388
quant_re_lu_7_58390
quant_conv2d_3_58393
quant_conv2d_3_58395
quant_conv2d_3_58397
quant_btch_norm_3_58400
quant_btch_norm_3_58402
quant_btch_norm_3_58404
quant_btch_norm_3_58406
quant_re_lu_8_58409
quant_re_lu_8_58411*
&quant_globalaveragepooling_layer_58414*
&quant_globalaveragepooling_layer_58416
quant_output_layer_58419
quant_output_layer_58421
quant_output_layer_58423
quant_output_layer_58425
quant_output_layer_58427
quant_output_layer_58429
identity’)quant_Btch_Norm-1/StatefulPartitionedCall’)quant_Btch_Norm-2/StatefulPartitionedCall’)quant_Btch_Norm-3/StatefulPartitionedCall’&quant_Conv2D-1/StatefulPartitionedCall’&quant_Conv2D-2/StatefulPartitionedCall’&quant_Conv2D-3/StatefulPartitionedCall’8quant_GlobalAveragePooling-Layer/StatefulPartitionedCall’*quant_Output-Layer/StatefulPartitionedCall’%quant_re_lu_6/StatefulPartitionedCall’%quant_re_lu_7/StatefulPartitionedCall’%quant_re_lu_8/StatefulPartitionedCall’&quantize_layer/StatefulPartitionedCall³
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_58346quantize_layer_58348*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quantize_layer_layer_call_and_return_conditional_losses_574182(
&quantize_layer/StatefulPartitionedCallφ
&quant_Conv2D-1/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_1_58351quant_conv2d_1_58353quant_conv2d_1_58355*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-1_layer_call_and_return_conditional_losses_574762(
&quant_Conv2D-1/StatefulPartitionedCall£
)quant_Btch_Norm-1/StatefulPartitionedCallStatefulPartitionedCall/quant_Conv2D-1/StatefulPartitionedCall:output:0quant_btch_norm_1_58358quant_btch_norm_1_58360quant_btch_norm_1_58362quant_btch_norm_1_58364*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-1_layer_call_and_return_conditional_losses_575392+
)quant_Btch_Norm-1/StatefulPartitionedCallΪ
%quant_re_lu_6/StatefulPartitionedCallStatefulPartitionedCall2quant_Btch_Norm-1/StatefulPartitionedCall:output:0quant_re_lu_6_58367quant_re_lu_6_58369*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_6_layer_call_and_return_conditional_losses_576202'
%quant_re_lu_6/StatefulPartitionedCallυ
&quant_Conv2D-2/StatefulPartitionedCallStatefulPartitionedCall.quant_re_lu_6/StatefulPartitionedCall:output:0quant_conv2d_2_58372quant_conv2d_2_58374quant_conv2d_2_58376*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-2_layer_call_and_return_conditional_losses_576792(
&quant_Conv2D-2/StatefulPartitionedCall£
)quant_Btch_Norm-2/StatefulPartitionedCallStatefulPartitionedCall/quant_Conv2D-2/StatefulPartitionedCall:output:0quant_btch_norm_2_58379quant_btch_norm_2_58381quant_btch_norm_2_58383quant_btch_norm_2_58385*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-2_layer_call_and_return_conditional_losses_577422+
)quant_Btch_Norm-2/StatefulPartitionedCallΪ
%quant_re_lu_7/StatefulPartitionedCallStatefulPartitionedCall2quant_Btch_Norm-2/StatefulPartitionedCall:output:0quant_re_lu_7_58388quant_re_lu_7_58390*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_7_layer_call_and_return_conditional_losses_578232'
%quant_re_lu_7/StatefulPartitionedCallυ
&quant_Conv2D-3/StatefulPartitionedCallStatefulPartitionedCall.quant_re_lu_7/StatefulPartitionedCall:output:0quant_conv2d_3_58393quant_conv2d_3_58395quant_conv2d_3_58397*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-3_layer_call_and_return_conditional_losses_578822(
&quant_Conv2D-3/StatefulPartitionedCall£
)quant_Btch_Norm-3/StatefulPartitionedCallStatefulPartitionedCall/quant_Conv2D-3/StatefulPartitionedCall:output:0quant_btch_norm_3_58400quant_btch_norm_3_58402quant_btch_norm_3_58404quant_btch_norm_3_58406*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-3_layer_call_and_return_conditional_losses_579452+
)quant_Btch_Norm-3/StatefulPartitionedCallΪ
%quant_re_lu_8/StatefulPartitionedCallStatefulPartitionedCall2quant_Btch_Norm-3/StatefulPartitionedCall:output:0quant_re_lu_8_58409quant_re_lu_8_58411*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_8_layer_call_and_return_conditional_losses_580262'
%quant_re_lu_8/StatefulPartitionedCall­
8quant_GlobalAveragePooling-Layer/StatefulPartitionedCallStatefulPartitionedCall.quant_re_lu_8/StatefulPartitionedCall:output:0&quant_globalaveragepooling_layer_58414&quant_globalaveragepooling_layer_58416*
Tin
2*
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
GPU 2J 8 *d
f_R]
[__inference_quant_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_580882:
8quant_GlobalAveragePooling-Layer/StatefulPartitionedCallκ
*quant_Output-Layer/StatefulPartitionedCallStatefulPartitionedCallAquant_GlobalAveragePooling-Layer/StatefulPartitionedCall:output:0quant_output_layer_58419quant_output_layer_58421quant_output_layer_58423quant_output_layer_58425quant_output_layer_58427quant_output_layer_58429*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_quant_Output-Layer_layer_call_and_return_conditional_losses_581802,
*quant_Output-Layer/StatefulPartitionedCall
IdentityIdentity3quant_Output-Layer/StatefulPartitionedCall:output:0*^quant_Btch_Norm-1/StatefulPartitionedCall*^quant_Btch_Norm-2/StatefulPartitionedCall*^quant_Btch_Norm-3/StatefulPartitionedCall'^quant_Conv2D-1/StatefulPartitionedCall'^quant_Conv2D-2/StatefulPartitionedCall'^quant_Conv2D-3/StatefulPartitionedCall9^quant_GlobalAveragePooling-Layer/StatefulPartitionedCall+^quant_Output-Layer/StatefulPartitionedCall&^quant_re_lu_6/StatefulPartitionedCall&^quant_re_lu_7/StatefulPartitionedCall&^quant_re_lu_8/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Δ
_input_shapes²
―:?????????1
:::::::::::::::::::::::::::::::::::::2V
)quant_Btch_Norm-1/StatefulPartitionedCall)quant_Btch_Norm-1/StatefulPartitionedCall2V
)quant_Btch_Norm-2/StatefulPartitionedCall)quant_Btch_Norm-2/StatefulPartitionedCall2V
)quant_Btch_Norm-3/StatefulPartitionedCall)quant_Btch_Norm-3/StatefulPartitionedCall2P
&quant_Conv2D-1/StatefulPartitionedCall&quant_Conv2D-1/StatefulPartitionedCall2P
&quant_Conv2D-2/StatefulPartitionedCall&quant_Conv2D-2/StatefulPartitionedCall2P
&quant_Conv2D-3/StatefulPartitionedCall&quant_Conv2D-3/StatefulPartitionedCall2t
8quant_GlobalAveragePooling-Layer/StatefulPartitionedCall8quant_GlobalAveragePooling-Layer/StatefulPartitionedCall2X
*quant_Output-Layer/StatefulPartitionedCall*quant_Output-Layer/StatefulPartitionedCall2N
%quant_re_lu_6/StatefulPartitionedCall%quant_re_lu_6/StatefulPartitionedCall2N
%quant_re_lu_7/StatefulPartitionedCall%quant_re_lu_7/StatefulPartitionedCall2N
%quant_re_lu_8/StatefulPartitionedCall%quant_re_lu_8/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs
R
Έ
M__inference_quant_Output-Layer_layer_call_and_return_conditional_losses_60000

inputs/
+lastvaluequant_rank_readvariableop_resource)
%lastvaluequant_assignminlast_resource)
%lastvaluequant_assignmaxlast_resource#
biasadd_readvariableop_resource:
6movingavgquantize_assignminema_readvariableop_resource:
6movingavgquantize_assignmaxema_readvariableop_resource
identity’LastValueQuant/AssignMaxLast’LastValueQuant/AssignMinLast’2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp’2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp΄
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:&*
dtype02$
"LastValueQuant/Rank/ReadVariableOpl
LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/Rankz
LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
LastValueQuant/range/startz
LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/range/deltaΉ
LastValueQuant/rangeRange#LastValueQuant/range/start:output:0LastValueQuant/Rank:output:0#LastValueQuant/range/delta:output:0*
_output_shapes
:2
LastValueQuant/rangeΌ
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:&*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOp©
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: 2
LastValueQuant/BatchMinΈ
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:&*
dtype02&
$LastValueQuant/Rank_1/ReadVariableOpp
LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/Rank_1~
LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
LastValueQuant/range_1/start~
LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/range_1/deltaΓ
LastValueQuant/range_1Range%LastValueQuant/range_1/start:output:0LastValueQuant/Rank_1:output:0%LastValueQuant/range_1/delta:output:0*
_output_shapes
:2
LastValueQuant/range_1Ό
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:&*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOp«
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/range_1:output:0*
T0*
_output_shapes
: 2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2
LastValueQuant/truediv/y‘
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: 2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: 2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: 2
LastValueQuant/MaximumΆ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastΆ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastΪ
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:&*
dtype027
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpο
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ο
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ΰ
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:&*
narrow_range(2(
&LastValueQuant/FakeQuantWithMinMaxVars
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayΝ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subΗ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul²
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayΝ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subΗ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul²
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1²
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars―
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&::::::2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp:O K
'
_output_shapes
:?????????&
 
_user_specified_nameinputs
Λ

L__inference_quant_Btch_Norm-1_layer_call_and_return_conditional_losses_57557

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&:::::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
Ψ

ψ
I__inference_quantize_layer_layer_call_and_return_conditional_losses_59342

inputsE
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resourceG
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource
identityξ
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpτ
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1°
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????1
2+
)AllValuesQuantize/FakeQuantWithMinMaxVars
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*/
_output_shapes
:?????????1
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????1
:::W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs
ζ#
±
I__inference_quant_Conv2D-1_layer_call_and_return_conditional_losses_59384

inputs3
/lastvaluequant_batchmin_readvariableop_resource)
%lastvaluequant_assignminlast_resource)
%lastvaluequant_assignmaxlast_resource
identity’LastValueQuant/AssignMaxLast’LastValueQuant/AssignMinLastΘ
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOp«
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesΒ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:&2
LastValueQuant/BatchMinΘ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOp«
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesΒ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:&2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2
LastValueQuant/truediv/y₯
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:&2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:&2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:&2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:&2
LastValueQuant/MaximumΆ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastΆ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastϊ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2€
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelΑ
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
Conv2D©
IdentityIdentityConv2D:output:0^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????1
:::2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast:W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs
₯

.__inference_quant_Conv2D-3_layer_call_fn_59759

inputs
unknown
	unknown_0
	unknown_1
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-3_layer_call_and_return_conditional_losses_578822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????&:::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
ζ#
±
I__inference_quant_Conv2D-1_layer_call_and_return_conditional_losses_57476

inputs3
/lastvaluequant_batchmin_readvariableop_resource)
%lastvaluequant_assignminlast_resource)
%lastvaluequant_assignmaxlast_resource
identity’LastValueQuant/AssignMaxLast’LastValueQuant/AssignMinLastΘ
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOp«
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesΒ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:&2
LastValueQuant/BatchMinΘ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOp«
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesΒ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:&2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2
LastValueQuant/truediv/y₯
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:&2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:&2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:&2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:&2
LastValueQuant/MaximumΆ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastΆ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastϊ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2€
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelΑ
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
Conv2D©
IdentityIdentityConv2D:output:0^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????1
:::2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast:W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs


+__inference_Btch_Norm-3_layer_call_fn_60246

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallͺ
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
GPU 2J 8 *O
fJRH
F__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_573692
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
Έ 
’
I__inference_quantize_layer_layer_call_and_return_conditional_losses_57418

inputs5
1allvaluesquantize_minimum_readvariableop_resource5
1allvaluesquantize_maximum_readvariableop_resource
identity’#AllValuesQuantize/AssignMaxAllValue’#AllValuesQuantize/AssignMinAllValue
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
AllValuesQuantize/Const
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/BatchMin
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
AllValuesQuantize/Const_1
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/BatchMaxΎ
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype02*
(AllValuesQuantize/Minimum/ReadVariableOpΉ
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Minimum
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
AllValuesQuantize/Minimum_1/y­
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Minimum_1Ύ
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype02*
(AllValuesQuantize/Maximum/ReadVariableOpΉ
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Maximum
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
AllValuesQuantize/Maximum_1/y­
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Maximum_1
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype02%
#AllValuesQuantize/AssignMinAllValue
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype02%
#AllValuesQuantize/AssignMaxAllValue
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype02:
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype02<
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1°
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????1
2+
)AllValuesQuantize/FakeQuantWithMinMaxVarsΫ
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue*
T0*/
_output_shapes
:?????????1
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????1
::2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue:W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs

?
F__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_60220

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&:::::i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
»
χ
H__inference_quant_re_lu_6_layer_call_and_return_conditional_losses_59518

inputsE
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resourceG
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????&2
Reluξ
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpτ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ό
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&:::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs

°
,__inference_sequential_4_layer_call_fn_58510
conv2d_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35
identity’StatefulPartitionedCallΖ
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
 #*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_584332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Δ
_input_shapes²
―:?????????1
:::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????1

(
_user_specified_nameConv2D-1_input

ρ
M__inference_quant_Output-Layer_layer_call_and_return_conditional_losses_60020

inputsB
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resourceD
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resourceD
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource#
biasadd_readvariableop_resourceE
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resourceG
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource
identityν
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:&*
dtype027
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpλ
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1λ
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ΰ
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:&*
narrow_range(2(
&LastValueQuant/FakeQuantWithMinMaxVars
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddξ
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpτ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1²
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&:::::::O K
'
_output_shapes
:?????????&
 
_user_specified_nameinputs
₯

.__inference_quant_Conv2D-1_layer_call_fn_59407

inputs
unknown
	unknown_0
	unknown_1
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-1_layer_call_and_return_conditional_losses_574762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????1
:::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs
R
Έ
M__inference_quant_Output-Layer_layer_call_and_return_conditional_losses_58180

inputs/
+lastvaluequant_rank_readvariableop_resource)
%lastvaluequant_assignminlast_resource)
%lastvaluequant_assignmaxlast_resource#
biasadd_readvariableop_resource:
6movingavgquantize_assignminema_readvariableop_resource:
6movingavgquantize_assignmaxema_readvariableop_resource
identity’LastValueQuant/AssignMaxLast’LastValueQuant/AssignMinLast’2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp’2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp΄
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:&*
dtype02$
"LastValueQuant/Rank/ReadVariableOpl
LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/Rankz
LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
LastValueQuant/range/startz
LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/range/deltaΉ
LastValueQuant/rangeRange#LastValueQuant/range/start:output:0LastValueQuant/Rank:output:0#LastValueQuant/range/delta:output:0*
_output_shapes
:2
LastValueQuant/rangeΌ
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:&*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOp©
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: 2
LastValueQuant/BatchMinΈ
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:&*
dtype02&
$LastValueQuant/Rank_1/ReadVariableOpp
LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/Rank_1~
LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
LastValueQuant/range_1/start~
LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/range_1/deltaΓ
LastValueQuant/range_1Range%LastValueQuant/range_1/start:output:0LastValueQuant/Rank_1:output:0%LastValueQuant/range_1/delta:output:0*
_output_shapes
:2
LastValueQuant/range_1Ό
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:&*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOp«
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/range_1:output:0*
T0*
_output_shapes
: 2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2
LastValueQuant/truediv/y‘
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: 2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: 2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: 2
LastValueQuant/MaximumΆ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastΆ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastΪ
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes

:&*
dtype027
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpο
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ο
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ΰ
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:&*
narrow_range(2(
&LastValueQuant/FakeQuantWithMinMaxVars
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayΝ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subΗ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul²
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayΝ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subΗ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul²
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1²
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars―
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&::::::2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp:O K
'
_output_shapes
:?????????&
 
_user_specified_nameinputs
Λ
€
1__inference_quant_Btch_Norm-2_layer_call_fn_59645

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-2_layer_call_and_return_conditional_losses_577422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
«)
Ι
H__inference_quant_re_lu_6_layer_call_and_return_conditional_losses_59508

inputs:
6movingavgquantize_assignminema_readvariableop_resource:
6movingavgquantize_assignmaxema_readvariableop_resource
identity’2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp’2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpV
ReluReluinputs*
T0*/
_output_shapes
:?????????&2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayΝ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subΗ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul²
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayΝ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subΗ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul²
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ό
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsω
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:03^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&::2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
₯

.__inference_quant_Conv2D-2_layer_call_fn_59583

inputs
unknown
	unknown_0
	unknown_1
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-2_layer_call_and_return_conditional_losses_576792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????&:::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs


-__inference_quant_re_lu_8_layer_call_fn_59879

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallό
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_8_layer_call_and_return_conditional_losses_580262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
»
χ
H__inference_quant_re_lu_8_layer_call_and_return_conditional_losses_58036

inputsE
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resourceG
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????&2
Reluξ
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpτ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ό
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&:::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
ζ#
±
I__inference_quant_Conv2D-3_layer_call_and_return_conditional_losses_57882

inputs3
/lastvaluequant_batchmin_readvariableop_resource)
%lastvaluequant_assignminlast_resource)
%lastvaluequant_assignmaxlast_resource
identity’LastValueQuant/AssignMaxLast’LastValueQuant/AssignMinLastΘ
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOp«
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesΒ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:&2
LastValueQuant/BatchMinΘ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOp«
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesΒ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:&2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2
LastValueQuant/truediv/y₯
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:&2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:&2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:&2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:&2
LastValueQuant/MaximumΆ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastΆ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastϊ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2€
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&&*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelΑ
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
Conv2D©
IdentityIdentityConv2D:output:0^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????&:::2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
»
χ
H__inference_quant_re_lu_6_layer_call_and_return_conditional_losses_57630

inputsE
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resourceG
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????&2
Reluξ
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpτ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ό
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&:::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
ό
©
L__inference_quant_Btch_Norm-3_layer_call_and_return_conditional_losses_59790

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ψ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
Ύ
£
F__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_60138

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o:*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
Ύ
£
F__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_57234

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o:*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
«)
Ι
H__inference_quant_re_lu_8_layer_call_and_return_conditional_losses_59860

inputs:
6movingavgquantize_assignminema_readvariableop_resource:
6movingavgquantize_assignmaxema_readvariableop_resource
identity’2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp’2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpV
ReluReluinputs*
T0*/
_output_shapes
:?????????&2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayΝ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subΗ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul²
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayΝ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subΗ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul²
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ό
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsω
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:03^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&::2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
«)
Ι
H__inference_quant_re_lu_6_layer_call_and_return_conditional_losses_57620

inputs:
6movingavgquantize_assignminema_readvariableop_resource:
6movingavgquantize_assignmaxema_readvariableop_resource
identity’2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp’2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpV
ReluReluinputs*
T0*/
_output_shapes
:?????????&2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayΝ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subΗ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul²
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayΝ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subΗ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul²
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ό
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsω
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:03^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&::2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
»
χ
H__inference_quant_re_lu_8_layer_call_and_return_conditional_losses_59870

inputsE
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resourceG
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????&2
Reluξ
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpτ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ό
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&:::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs

¨
,__inference_sequential_4_layer_call_fn_59312

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35
identity’StatefulPartitionedCallΨ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*G
_read_only_resource_inputs)
'%	
 !"#$%*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_586022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Δ
_input_shapes²
―:?????????1
:::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs
Ύ
£
F__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_57338

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o:*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
Λ
€
1__inference_quant_Btch_Norm-3_layer_call_fn_59821

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-3_layer_call_and_return_conditional_losses_579452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs


+__inference_Btch_Norm-2_layer_call_fn_60182

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCallͺ
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
GPU 2J 8 *O
fJRH
F__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_572652
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
ς
§
#__inference_signature_wrapper_58768
conv2d_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35
identity’StatefulPartitionedCallΉ
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*G
_read_only_resource_inputs)
'%	
 !"#$%*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_570682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Δ
_input_shapes²
―:?????????1
:::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????1

(
_user_specified_nameConv2D-1_input


.__inference_quantize_layer_layer_call_fn_59351

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallύ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quantize_layer_layer_call_and_return_conditional_losses_574182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????1
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????1
::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs
ζ#
±
I__inference_quant_Conv2D-2_layer_call_and_return_conditional_losses_57679

inputs3
/lastvaluequant_batchmin_readvariableop_resource)
%lastvaluequant_assignminlast_resource)
%lastvaluequant_assignmaxlast_resource
identity’LastValueQuant/AssignMaxLast’LastValueQuant/AssignMinLastΘ
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOp«
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesΒ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:&2
LastValueQuant/BatchMinΘ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOp«
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesΒ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:&2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2
LastValueQuant/truediv/y₯
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:&2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:&2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:&2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:&2
LastValueQuant/MaximumΆ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastΆ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastϊ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2€
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&&*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelΑ
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
Conv2D©
IdentityIdentityConv2D:output:0^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????&:::2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs

Φ
I__inference_quant_Conv2D-1_layer_call_and_return_conditional_losses_59396

inputsL
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resourceN
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resourceN
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource
identity
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:&*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2€
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelΑ
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
Conv2Dk
IdentityIdentityConv2D:output:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????1
::::W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs
Ν
€
1__inference_quant_Btch_Norm-3_layer_call_fn_59834

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-3_layer_call_and_return_conditional_losses_579632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs


+__inference_Btch_Norm-3_layer_call_fn_60233

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall¨
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
GPU 2J 8 *O
fJRH
F__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_573382
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs


+__inference_Btch_Norm-2_layer_call_fn_60169

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall¨
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
GPU 2J 8 *O
fJRH
F__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_572342
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs

Φ
I__inference_quant_Conv2D-3_layer_call_and_return_conditional_losses_59748

inputsL
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resourceN
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resourceN
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource
identity
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:&&*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2€
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&&*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelΑ
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
Conv2Dk
IdentityIdentityConv2D:output:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????&::::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs

?
F__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_57369

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&:::::i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
βΑ

 __inference__wrapped_model_57068
conv2d_1_inputa
]sequential_4_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resourcec
_sequential_4_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resourceh
dsequential_4_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resourcej
fsequential_4_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resourcej
fsequential_4_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:
6sequential_4_quant_btch_norm_1_readvariableop_resource<
8sequential_4_quant_btch_norm_1_readvariableop_1_resourceK
Gsequential_4_quant_btch_norm_1_fusedbatchnormv3_readvariableop_resourceM
Isequential_4_quant_btch_norm_1_fusedbatchnormv3_readvariableop_1_resource`
\sequential_4_quant_re_lu_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resourceb
^sequential_4_quant_re_lu_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resourceh
dsequential_4_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resourcej
fsequential_4_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resourcej
fsequential_4_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:
6sequential_4_quant_btch_norm_2_readvariableop_resource<
8sequential_4_quant_btch_norm_2_readvariableop_1_resourceK
Gsequential_4_quant_btch_norm_2_fusedbatchnormv3_readvariableop_resourceM
Isequential_4_quant_btch_norm_2_fusedbatchnormv3_readvariableop_1_resource`
\sequential_4_quant_re_lu_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_resourceb
^sequential_4_quant_re_lu_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resourceh
dsequential_4_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resourcej
fsequential_4_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resourcej
fsequential_4_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:
6sequential_4_quant_btch_norm_3_readvariableop_resource<
8sequential_4_quant_btch_norm_3_readvariableop_1_resourceK
Gsequential_4_quant_btch_norm_3_fusedbatchnormv3_readvariableop_resourceM
Isequential_4_quant_btch_norm_3_fusedbatchnormv3_readvariableop_1_resource`
\sequential_4_quant_re_lu_8_movingavgquantize_fakequantwithminmaxvars_readvariableop_resourceb
^sequential_4_quant_re_lu_8_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resources
osequential_4_quant_globalaveragepooling_layer_movingavgquantize_fakequantwithminmaxvars_readvariableop_resourceu
qsequential_4_quant_globalaveragepooling_layer_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resourceb
^sequential_4_quant_output_layer_lastvaluequant_fakequantwithminmaxvars_readvariableop_resourced
`sequential_4_quant_output_layer_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resourced
`sequential_4_quant_output_layer_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resourceC
?sequential_4_quant_output_layer_biasadd_readvariableop_resourcee
asequential_4_quant_output_layer_movingavgquantize_fakequantwithminmaxvars_readvariableop_resourceg
csequential_4_quant_output_layer_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource
identityΒ
Tsequential_4/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp]sequential_4_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02V
Tsequential_4/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpΘ
Vsequential_4/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp_sequential_4_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02X
Vsequential_4/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1¨
Esequential_4/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsconv2d_1_input\sequential_4/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0^sequential_4/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????1
2G
Esequential_4/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsη
[sequential_4/quant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpdsequential_4_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:&*
dtype02]
[sequential_4/quant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpα
]sequential_4/quant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpfsequential_4_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:&*
dtype02_
]sequential_4/quant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1α
]sequential_4/quant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpfsequential_4_quant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:&*
dtype02_
]sequential_4/quant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2°
Lsequential_4/quant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelcsequential_4/quant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0esequential_4/quant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0esequential_4/quant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&*
narrow_range(2N
Lsequential_4/quant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelή
"sequential_4/quant_Conv2D-1/Conv2DConv2DOsequential_4/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Vsequential_4/quant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2$
"sequential_4/quant_Conv2D-1/Conv2DΡ
-sequential_4/quant_Btch_Norm-1/ReadVariableOpReadVariableOp6sequential_4_quant_btch_norm_1_readvariableop_resource*
_output_shapes
:&*
dtype02/
-sequential_4/quant_Btch_Norm-1/ReadVariableOpΧ
/sequential_4/quant_Btch_Norm-1/ReadVariableOp_1ReadVariableOp8sequential_4_quant_btch_norm_1_readvariableop_1_resource*
_output_shapes
:&*
dtype021
/sequential_4/quant_Btch_Norm-1/ReadVariableOp_1
>sequential_4/quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_4_quant_btch_norm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02@
>sequential_4/quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp
@sequential_4/quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_4_quant_btch_norm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02B
@sequential_4/quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1©
/sequential_4/quant_Btch_Norm-1/FusedBatchNormV3FusedBatchNormV3+sequential_4/quant_Conv2D-1/Conv2D:output:05sequential_4/quant_Btch_Norm-1/ReadVariableOp:value:07sequential_4/quant_Btch_Norm-1/ReadVariableOp_1:value:0Fsequential_4/quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp:value:0Hsequential_4/quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
is_training( 21
/sequential_4/quant_Btch_Norm-1/FusedBatchNormV3Ή
sequential_4/quant_re_lu_6/ReluRelu3sequential_4/quant_Btch_Norm-1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&2!
sequential_4/quant_re_lu_6/ReluΏ
Ssequential_4/quant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp\sequential_4_quant_re_lu_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02U
Ssequential_4/quant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpΕ
Usequential_4/quant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp^sequential_4_quant_re_lu_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02W
Usequential_4/quant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Γ
Dsequential_4/quant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars-sequential_4/quant_re_lu_6/Relu:activations:0[sequential_4/quant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0]sequential_4/quant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&2F
Dsequential_4/quant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsη
[sequential_4/quant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpdsequential_4_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:&&*
dtype02]
[sequential_4/quant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpα
]sequential_4/quant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpfsequential_4_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:&*
dtype02_
]sequential_4/quant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1α
]sequential_4/quant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpfsequential_4_quant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:&*
dtype02_
]sequential_4/quant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2°
Lsequential_4/quant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelcsequential_4/quant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0esequential_4/quant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0esequential_4/quant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&&*
narrow_range(2N
Lsequential_4/quant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelέ
"sequential_4/quant_Conv2D-2/Conv2DConv2DNsequential_4/quant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Vsequential_4/quant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2$
"sequential_4/quant_Conv2D-2/Conv2DΡ
-sequential_4/quant_Btch_Norm-2/ReadVariableOpReadVariableOp6sequential_4_quant_btch_norm_2_readvariableop_resource*
_output_shapes
:&*
dtype02/
-sequential_4/quant_Btch_Norm-2/ReadVariableOpΧ
/sequential_4/quant_Btch_Norm-2/ReadVariableOp_1ReadVariableOp8sequential_4_quant_btch_norm_2_readvariableop_1_resource*
_output_shapes
:&*
dtype021
/sequential_4/quant_Btch_Norm-2/ReadVariableOp_1
>sequential_4/quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_4_quant_btch_norm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02@
>sequential_4/quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp
@sequential_4/quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_4_quant_btch_norm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02B
@sequential_4/quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1©
/sequential_4/quant_Btch_Norm-2/FusedBatchNormV3FusedBatchNormV3+sequential_4/quant_Conv2D-2/Conv2D:output:05sequential_4/quant_Btch_Norm-2/ReadVariableOp:value:07sequential_4/quant_Btch_Norm-2/ReadVariableOp_1:value:0Fsequential_4/quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp:value:0Hsequential_4/quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
is_training( 21
/sequential_4/quant_Btch_Norm-2/FusedBatchNormV3Ή
sequential_4/quant_re_lu_7/ReluRelu3sequential_4/quant_Btch_Norm-2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&2!
sequential_4/quant_re_lu_7/ReluΏ
Ssequential_4/quant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp\sequential_4_quant_re_lu_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02U
Ssequential_4/quant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpΕ
Usequential_4/quant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp^sequential_4_quant_re_lu_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02W
Usequential_4/quant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Γ
Dsequential_4/quant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars-sequential_4/quant_re_lu_7/Relu:activations:0[sequential_4/quant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0]sequential_4/quant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&2F
Dsequential_4/quant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVarsη
[sequential_4/quant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpdsequential_4_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:&&*
dtype02]
[sequential_4/quant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpα
]sequential_4/quant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpfsequential_4_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:&*
dtype02_
]sequential_4/quant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1α
]sequential_4/quant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpfsequential_4_quant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:&*
dtype02_
]sequential_4/quant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2°
Lsequential_4/quant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelcsequential_4/quant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0esequential_4/quant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0esequential_4/quant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&&*
narrow_range(2N
Lsequential_4/quant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelέ
"sequential_4/quant_Conv2D-3/Conv2DConv2DNsequential_4/quant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Vsequential_4/quant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2$
"sequential_4/quant_Conv2D-3/Conv2DΡ
-sequential_4/quant_Btch_Norm-3/ReadVariableOpReadVariableOp6sequential_4_quant_btch_norm_3_readvariableop_resource*
_output_shapes
:&*
dtype02/
-sequential_4/quant_Btch_Norm-3/ReadVariableOpΧ
/sequential_4/quant_Btch_Norm-3/ReadVariableOp_1ReadVariableOp8sequential_4_quant_btch_norm_3_readvariableop_1_resource*
_output_shapes
:&*
dtype021
/sequential_4/quant_Btch_Norm-3/ReadVariableOp_1
>sequential_4/quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_4_quant_btch_norm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02@
>sequential_4/quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp
@sequential_4/quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_4_quant_btch_norm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02B
@sequential_4/quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1©
/sequential_4/quant_Btch_Norm-3/FusedBatchNormV3FusedBatchNormV3+sequential_4/quant_Conv2D-3/Conv2D:output:05sequential_4/quant_Btch_Norm-3/ReadVariableOp:value:07sequential_4/quant_Btch_Norm-3/ReadVariableOp_1:value:0Fsequential_4/quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp:value:0Hsequential_4/quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
is_training( 21
/sequential_4/quant_Btch_Norm-3/FusedBatchNormV3Ή
sequential_4/quant_re_lu_8/ReluRelu3sequential_4/quant_Btch_Norm-3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&2!
sequential_4/quant_re_lu_8/ReluΏ
Ssequential_4/quant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp\sequential_4_quant_re_lu_8_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02U
Ssequential_4/quant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpΕ
Usequential_4/quant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp^sequential_4_quant_re_lu_8_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02W
Usequential_4/quant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Γ
Dsequential_4/quant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars-sequential_4/quant_re_lu_8/Relu:activations:0[sequential_4/quant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0]sequential_4/quant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&2F
Dsequential_4/quant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVarsέ
Dsequential_4/quant_GlobalAveragePooling-Layer/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2F
Dsequential_4/quant_GlobalAveragePooling-Layer/Mean/reduction_indicesΑ
2sequential_4/quant_GlobalAveragePooling-Layer/MeanMeanNsequential_4/quant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Msequential_4/quant_GlobalAveragePooling-Layer/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????&24
2sequential_4/quant_GlobalAveragePooling-Layer/Meanψ
fsequential_4/quant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOposequential_4_quant_globalaveragepooling_layer_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02h
fsequential_4/quant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpώ
hsequential_4/quant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpqsequential_4_quant_globalaveragepooling_layer_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02j
hsequential_4/quant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
Wsequential_4/quant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars;sequential_4/quant_GlobalAveragePooling-Layer/Mean:output:0nsequential_4/quant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0psequential_4/quant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????&2Y
Wsequential_4/quant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVarsΝ
Usequential_4/quant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp^sequential_4_quant_output_layer_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:&*
dtype02W
Usequential_4/quant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpΛ
Wsequential_4/quant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp`sequential_4_quant_output_layer_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02Y
Wsequential_4/quant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Λ
Wsequential_4/quant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp`sequential_4_quant_output_layer_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype02Y
Wsequential_4/quant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2
Fsequential_4/quant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars]sequential_4/quant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0_sequential_4/quant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0_sequential_4/quant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:&*
narrow_range(2H
Fsequential_4/quant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVarsΑ
&sequential_4/quant_Output-Layer/MatMulMatMulasequential_4/quant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Psequential_4/quant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2(
&sequential_4/quant_Output-Layer/MatMulμ
6sequential_4/quant_Output-Layer/BiasAdd/ReadVariableOpReadVariableOp?sequential_4_quant_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential_4/quant_Output-Layer/BiasAdd/ReadVariableOp
'sequential_4/quant_Output-Layer/BiasAddBiasAdd0sequential_4/quant_Output-Layer/MatMul:product:0>sequential_4/quant_Output-Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'sequential_4/quant_Output-Layer/BiasAddΞ
Xsequential_4/quant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpasequential_4_quant_output_layer_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02Z
Xsequential_4/quant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpΤ
Zsequential_4/quant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpcsequential_4_quant_output_layer_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02\
Zsequential_4/quant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
Isequential_4/quant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars0sequential_4/quant_Output-Layer/BiasAdd:output:0`sequential_4/quant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0bsequential_4/quant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????2K
Isequential_4/quant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars§
IdentityIdentitySsequential_4/quant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Δ
_input_shapes²
―:?????????1
::::::::::::::::::::::::::::::::::::::_ [
/
_output_shapes
:?????????1

(
_user_specified_nameConv2D-1_input
Λ

L__inference_quant_Btch_Norm-2_layer_call_and_return_conditional_losses_59632

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&:::::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
Ύ
£
F__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_60074

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o:*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs

V
:__inference_GlobalAveragePooling-Layer_layer_call_fn_57393

inputs
identityά
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
GPU 2J 8 *^
fYRW
U__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_573872
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
ό
©
L__inference_quant_Btch_Norm-3_layer_call_and_return_conditional_losses_57945

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ψ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs

Φ
I__inference_quant_Conv2D-1_layer_call_and_return_conditional_losses_57488

inputsL
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resourceN
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resourceN
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource
identity
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:&*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2€
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelΑ
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
Conv2Dk
IdentityIdentityConv2D:output:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????1
::::W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs
Φ

[__inference_quant_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_58099

inputsE
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resourceG
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indiceso
MeanMeaninputsMean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????&2
Meanξ
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpτ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1―
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsMean:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????&2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&:::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
ο
Γ
2__inference_quant_Output-Layer_layer_call_fn_60054

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity’StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_quant_Output-Layer_layer_call_and_return_conditional_losses_582002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????&
 
_user_specified_nameinputs
Ύ
£
F__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_60202

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o:*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs

Τ
G__inference_sequential_4_layer_call_and_return_conditional_losses_59038

inputsD
@quantize_layer_allvaluesquantize_minimum_readvariableop_resourceD
@quantize_layer_allvaluesquantize_maximum_readvariableop_resourceB
>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource8
4quant_conv2d_1_lastvaluequant_assignminlast_resource8
4quant_conv2d_1_lastvaluequant_assignmaxlast_resource-
)quant_btch_norm_1_readvariableop_resource/
+quant_btch_norm_1_readvariableop_1_resource>
:quant_btch_norm_1_fusedbatchnormv3_readvariableop_resource@
<quant_btch_norm_1_fusedbatchnormv3_readvariableop_1_resourceH
Dquant_re_lu_6_movingavgquantize_assignminema_readvariableop_resourceH
Dquant_re_lu_6_movingavgquantize_assignmaxema_readvariableop_resourceB
>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource8
4quant_conv2d_2_lastvaluequant_assignminlast_resource8
4quant_conv2d_2_lastvaluequant_assignmaxlast_resource-
)quant_btch_norm_2_readvariableop_resource/
+quant_btch_norm_2_readvariableop_1_resource>
:quant_btch_norm_2_fusedbatchnormv3_readvariableop_resource@
<quant_btch_norm_2_fusedbatchnormv3_readvariableop_1_resourceH
Dquant_re_lu_7_movingavgquantize_assignminema_readvariableop_resourceH
Dquant_re_lu_7_movingavgquantize_assignmaxema_readvariableop_resourceB
>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource8
4quant_conv2d_3_lastvaluequant_assignminlast_resource8
4quant_conv2d_3_lastvaluequant_assignmaxlast_resource-
)quant_btch_norm_3_readvariableop_resource/
+quant_btch_norm_3_readvariableop_1_resource>
:quant_btch_norm_3_fusedbatchnormv3_readvariableop_resource@
<quant_btch_norm_3_fusedbatchnormv3_readvariableop_1_resourceH
Dquant_re_lu_8_movingavgquantize_assignminema_readvariableop_resourceH
Dquant_re_lu_8_movingavgquantize_assignmaxema_readvariableop_resource[
Wquant_globalaveragepooling_layer_movingavgquantize_assignminema_readvariableop_resource[
Wquant_globalaveragepooling_layer_movingavgquantize_assignmaxema_readvariableop_resourceB
>quant_output_layer_lastvaluequant_rank_readvariableop_resource<
8quant_output_layer_lastvaluequant_assignminlast_resource<
8quant_output_layer_lastvaluequant_assignmaxlast_resource6
2quant_output_layer_biasadd_readvariableop_resourceM
Iquant_output_layer_movingavgquantize_assignminema_readvariableop_resourceM
Iquant_output_layer_movingavgquantize_assignmaxema_readvariableop_resource
identity’ quant_Btch_Norm-1/AssignNewValue’"quant_Btch_Norm-1/AssignNewValue_1’ quant_Btch_Norm-2/AssignNewValue’"quant_Btch_Norm-2/AssignNewValue_1’ quant_Btch_Norm-3/AssignNewValue’"quant_Btch_Norm-3/AssignNewValue_1’+quant_Conv2D-1/LastValueQuant/AssignMaxLast’+quant_Conv2D-1/LastValueQuant/AssignMinLast’+quant_Conv2D-2/LastValueQuant/AssignMaxLast’+quant_Conv2D-2/LastValueQuant/AssignMinLast’+quant_Conv2D-3/LastValueQuant/AssignMaxLast’+quant_Conv2D-3/LastValueQuant/AssignMinLast’Squant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp’Squant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp’/quant_Output-Layer/LastValueQuant/AssignMaxLast’/quant_Output-Layer/LastValueQuant/AssignMinLast’Equant_Output-Layer/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp’Equant_Output-Layer/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp’@quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp’@quant_re_lu_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp’@quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp’@quant_re_lu_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp’@quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp’@quant_re_lu_8/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp’2quantize_layer/AllValuesQuantize/AssignMaxAllValue’2quantize_layer/AllValuesQuantize/AssignMinAllValue©
&quantize_layer/AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&quantize_layer/AllValuesQuantize/Const·
)quantize_layer/AllValuesQuantize/BatchMinMininputs/quantize_layer/AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: 2+
)quantize_layer/AllValuesQuantize/BatchMin­
(quantize_layer/AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2*
(quantize_layer/AllValuesQuantize/Const_1Ή
)quantize_layer/AllValuesQuantize/BatchMaxMaxinputs1quantize_layer/AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: 2+
)quantize_layer/AllValuesQuantize/BatchMaxλ
7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype029
7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOpυ
(quantize_layer/AllValuesQuantize/MinimumMinimum?quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp:value:02quantize_layer/AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: 2*
(quantize_layer/AllValuesQuantize/Minimum‘
,quantize_layer/AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,quantize_layer/AllValuesQuantize/Minimum_1/yι
*quantize_layer/AllValuesQuantize/Minimum_1Minimum,quantize_layer/AllValuesQuantize/Minimum:z:05quantize_layer/AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: 2,
*quantize_layer/AllValuesQuantize/Minimum_1λ
7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype029
7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOpυ
(quantize_layer/AllValuesQuantize/MaximumMaximum?quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp:value:02quantize_layer/AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: 2*
(quantize_layer/AllValuesQuantize/Maximum‘
,quantize_layer/AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,quantize_layer/AllValuesQuantize/Maximum_1/yι
*quantize_layer/AllValuesQuantize/Maximum_1Maximum,quantize_layer/AllValuesQuantize/Maximum:z:05quantize_layer/AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: 2,
*quantize_layer/AllValuesQuantize/Maximum_1Λ
2quantize_layer/AllValuesQuantize/AssignMinAllValueAssignVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource.quantize_layer/AllValuesQuantize/Minimum_1:z:08^quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype024
2quantize_layer/AllValuesQuantize/AssignMinAllValueΛ
2quantize_layer/AllValuesQuantize/AssignMaxAllValueAssignVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource.quantize_layer/AllValuesQuantize/Maximum_1:z:08^quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype024
2quantize_layer/AllValuesQuantize/AssignMaxAllValueΐ
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource3^quantize_layer/AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype02I
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpΔ
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource3^quantize_layer/AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype02K
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1μ
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsOquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????1
2:
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsυ
5quant_Conv2D-1/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&*
dtype027
5quant_Conv2D-1/LastValueQuant/BatchMin/ReadVariableOpΙ
8quant_Conv2D-1/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_Conv2D-1/LastValueQuant/BatchMin/reduction_indicesώ
&quant_Conv2D-1/LastValueQuant/BatchMinMin=quant_Conv2D-1/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_Conv2D-1/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:&2(
&quant_Conv2D-1/LastValueQuant/BatchMinυ
5quant_Conv2D-1/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&*
dtype027
5quant_Conv2D-1/LastValueQuant/BatchMax/ReadVariableOpΙ
8quant_Conv2D-1/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_Conv2D-1/LastValueQuant/BatchMax/reduction_indicesώ
&quant_Conv2D-1/LastValueQuant/BatchMaxMax=quant_Conv2D-1/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_Conv2D-1/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:&2(
&quant_Conv2D-1/LastValueQuant/BatchMax
'quant_Conv2D-1/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2)
'quant_Conv2D-1/LastValueQuant/truediv/yα
%quant_Conv2D-1/LastValueQuant/truedivRealDiv/quant_Conv2D-1/LastValueQuant/BatchMax:output:00quant_Conv2D-1/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:&2'
%quant_Conv2D-1/LastValueQuant/truedivΪ
%quant_Conv2D-1/LastValueQuant/MinimumMinimum/quant_Conv2D-1/LastValueQuant/BatchMin:output:0)quant_Conv2D-1/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:&2'
%quant_Conv2D-1/LastValueQuant/Minimum
#quant_Conv2D-1/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2%
#quant_Conv2D-1/LastValueQuant/mul/yΡ
!quant_Conv2D-1/LastValueQuant/mulMul/quant_Conv2D-1/LastValueQuant/BatchMin:output:0,quant_Conv2D-1/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:&2#
!quant_Conv2D-1/LastValueQuant/mulΦ
%quant_Conv2D-1/LastValueQuant/MaximumMaximum/quant_Conv2D-1/LastValueQuant/BatchMax:output:0%quant_Conv2D-1/LastValueQuant/mul:z:0*
T0*
_output_shapes
:&2'
%quant_Conv2D-1/LastValueQuant/Maximumς
+quant_Conv2D-1/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_1_lastvaluequant_assignminlast_resource)quant_Conv2D-1/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02-
+quant_Conv2D-1/LastValueQuant/AssignMinLastς
+quant_Conv2D-1/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_1_lastvaluequant_assignmaxlast_resource)quant_Conv2D-1/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02-
+quant_Conv2D-1/LastValueQuant/AssignMaxLast§
Nquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_1_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&*
dtype02P
Nquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpΓ
Pquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_1_lastvaluequant_assignminlast_resource,^quant_Conv2D-1/LastValueQuant/AssignMinLast*
_output_shapes
:&*
dtype02R
Pquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Γ
Pquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_1_lastvaluequant_assignmaxlast_resource,^quant_Conv2D-1/LastValueQuant/AssignMaxLast*
_output_shapes
:&*
dtype02R
Pquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ο
?quant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&*
narrow_range(2A
?quant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelͺ
quant_Conv2D-1/Conv2DConv2DBquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
quant_Conv2D-1/Conv2Dͺ
 quant_Btch_Norm-1/ReadVariableOpReadVariableOp)quant_btch_norm_1_readvariableop_resource*
_output_shapes
:&*
dtype02"
 quant_Btch_Norm-1/ReadVariableOp°
"quant_Btch_Norm-1/ReadVariableOp_1ReadVariableOp+quant_btch_norm_1_readvariableop_1_resource*
_output_shapes
:&*
dtype02$
"quant_Btch_Norm-1/ReadVariableOp_1έ
1quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOpReadVariableOp:quant_btch_norm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype023
1quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOpγ
3quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<quant_btch_norm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype025
3quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1ά
"quant_Btch_Norm-1/FusedBatchNormV3FusedBatchNormV3quant_Conv2D-1/Conv2D:output:0(quant_Btch_Norm-1/ReadVariableOp:value:0*quant_Btch_Norm-1/ReadVariableOp_1:value:09quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp:value:0;quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
exponential_avg_factor%fff?2$
"quant_Btch_Norm-1/FusedBatchNormV3λ
 quant_Btch_Norm-1/AssignNewValueAssignVariableOp:quant_btch_norm_1_fusedbatchnormv3_readvariableop_resource/quant_Btch_Norm-1/FusedBatchNormV3:batch_mean:02^quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp*M
_classC
A?loc:@quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02"
 quant_Btch_Norm-1/AssignNewValueω
"quant_Btch_Norm-1/AssignNewValue_1AssignVariableOp<quant_btch_norm_1_fusedbatchnormv3_readvariableop_1_resource3quant_Btch_Norm-1/FusedBatchNormV3:batch_variance:04^quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1*O
_classE
CAloc:@quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02$
"quant_Btch_Norm-1/AssignNewValue_1
quant_re_lu_6/ReluRelu&quant_Btch_Norm-1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&2
quant_re_lu_6/Relu§
%quant_re_lu_6/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%quant_re_lu_6/MovingAvgQuantize/ConstΞ
(quant_re_lu_6/MovingAvgQuantize/BatchMinMin quant_re_lu_6/Relu:activations:0.quant_re_lu_6/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2*
(quant_re_lu_6/MovingAvgQuantize/BatchMin«
'quant_re_lu_6/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2)
'quant_re_lu_6/MovingAvgQuantize/Const_1Π
(quant_re_lu_6/MovingAvgQuantize/BatchMaxMax quant_re_lu_6/Relu:activations:00quant_re_lu_6/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2*
(quant_re_lu_6/MovingAvgQuantize/BatchMax
)quant_re_lu_6/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)quant_re_lu_6/MovingAvgQuantize/Minimum/yε
'quant_re_lu_6/MovingAvgQuantize/MinimumMinimum1quant_re_lu_6/MovingAvgQuantize/BatchMin:output:02quant_re_lu_6/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2)
'quant_re_lu_6/MovingAvgQuantize/Minimum
)quant_re_lu_6/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)quant_re_lu_6/MovingAvgQuantize/Maximum/yε
'quant_re_lu_6/MovingAvgQuantize/MaximumMaximum1quant_re_lu_6/MovingAvgQuantize/BatchMax:output:02quant_re_lu_6/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2)
'quant_re_lu_6/MovingAvgQuantize/Maximum­
2quant_re_lu_6/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:24
2quant_re_lu_6/MovingAvgQuantize/AssignMinEma/decayχ
;quant_re_lu_6/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpDquant_re_lu_6_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02=
;quant_re_lu_6/MovingAvgQuantize/AssignMinEma/ReadVariableOpώ
0quant_re_lu_6/MovingAvgQuantize/AssignMinEma/subSubCquant_re_lu_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0+quant_re_lu_6/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 22
0quant_re_lu_6/MovingAvgQuantize/AssignMinEma/sub?
0quant_re_lu_6/MovingAvgQuantize/AssignMinEma/mulMul4quant_re_lu_6/MovingAvgQuantize/AssignMinEma/sub:z:0;quant_re_lu_6/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 22
0quant_re_lu_6/MovingAvgQuantize/AssignMinEma/mulψ
@quant_re_lu_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpDquant_re_lu_6_movingavgquantize_assignminema_readvariableop_resource4quant_re_lu_6/MovingAvgQuantize/AssignMinEma/mul:z:0<^quant_re_lu_6/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02B
@quant_re_lu_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp­
2quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:24
2quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/decayχ
;quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpDquant_re_lu_6_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02=
;quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOpώ
0quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/subSubCquant_re_lu_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0+quant_re_lu_6/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 22
0quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/sub?
0quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/mulMul4quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/sub:z:0;quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 22
0quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/mulψ
@quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpDquant_re_lu_6_movingavgquantize_assignmaxema_readvariableop_resource4quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/mul:z:0<^quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02B
@quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpΠ
Fquant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpDquant_re_lu_6_movingavgquantize_assignminema_readvariableop_resourceA^quant_re_lu_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02H
Fquant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpΤ
Hquant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpDquant_re_lu_6_movingavgquantize_assignmaxema_readvariableop_resourceA^quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02J
Hquant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
7quant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars quant_re_lu_6/Relu:activations:0Nquant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&29
7quant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsυ
5quant_Conv2D-2/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype027
5quant_Conv2D-2/LastValueQuant/BatchMin/ReadVariableOpΙ
8quant_Conv2D-2/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_Conv2D-2/LastValueQuant/BatchMin/reduction_indicesώ
&quant_Conv2D-2/LastValueQuant/BatchMinMin=quant_Conv2D-2/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_Conv2D-2/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:&2(
&quant_Conv2D-2/LastValueQuant/BatchMinυ
5quant_Conv2D-2/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype027
5quant_Conv2D-2/LastValueQuant/BatchMax/ReadVariableOpΙ
8quant_Conv2D-2/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_Conv2D-2/LastValueQuant/BatchMax/reduction_indicesώ
&quant_Conv2D-2/LastValueQuant/BatchMaxMax=quant_Conv2D-2/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_Conv2D-2/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:&2(
&quant_Conv2D-2/LastValueQuant/BatchMax
'quant_Conv2D-2/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2)
'quant_Conv2D-2/LastValueQuant/truediv/yα
%quant_Conv2D-2/LastValueQuant/truedivRealDiv/quant_Conv2D-2/LastValueQuant/BatchMax:output:00quant_Conv2D-2/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:&2'
%quant_Conv2D-2/LastValueQuant/truedivΪ
%quant_Conv2D-2/LastValueQuant/MinimumMinimum/quant_Conv2D-2/LastValueQuant/BatchMin:output:0)quant_Conv2D-2/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:&2'
%quant_Conv2D-2/LastValueQuant/Minimum
#quant_Conv2D-2/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2%
#quant_Conv2D-2/LastValueQuant/mul/yΡ
!quant_Conv2D-2/LastValueQuant/mulMul/quant_Conv2D-2/LastValueQuant/BatchMin:output:0,quant_Conv2D-2/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:&2#
!quant_Conv2D-2/LastValueQuant/mulΦ
%quant_Conv2D-2/LastValueQuant/MaximumMaximum/quant_Conv2D-2/LastValueQuant/BatchMax:output:0%quant_Conv2D-2/LastValueQuant/mul:z:0*
T0*
_output_shapes
:&2'
%quant_Conv2D-2/LastValueQuant/Maximumς
+quant_Conv2D-2/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_2_lastvaluequant_assignminlast_resource)quant_Conv2D-2/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02-
+quant_Conv2D-2/LastValueQuant/AssignMinLastς
+quant_Conv2D-2/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_2_lastvaluequant_assignmaxlast_resource)quant_Conv2D-2/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02-
+quant_Conv2D-2/LastValueQuant/AssignMaxLast§
Nquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_2_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype02P
Nquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpΓ
Pquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_2_lastvaluequant_assignminlast_resource,^quant_Conv2D-2/LastValueQuant/AssignMinLast*
_output_shapes
:&*
dtype02R
Pquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Γ
Pquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_2_lastvaluequant_assignmaxlast_resource,^quant_Conv2D-2/LastValueQuant/AssignMaxLast*
_output_shapes
:&*
dtype02R
Pquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ο
?quant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&&*
narrow_range(2A
?quant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel©
quant_Conv2D-2/Conv2DConv2DAquant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
quant_Conv2D-2/Conv2Dͺ
 quant_Btch_Norm-2/ReadVariableOpReadVariableOp)quant_btch_norm_2_readvariableop_resource*
_output_shapes
:&*
dtype02"
 quant_Btch_Norm-2/ReadVariableOp°
"quant_Btch_Norm-2/ReadVariableOp_1ReadVariableOp+quant_btch_norm_2_readvariableop_1_resource*
_output_shapes
:&*
dtype02$
"quant_Btch_Norm-2/ReadVariableOp_1έ
1quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOpReadVariableOp:quant_btch_norm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype023
1quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOpγ
3quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<quant_btch_norm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype025
3quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1ά
"quant_Btch_Norm-2/FusedBatchNormV3FusedBatchNormV3quant_Conv2D-2/Conv2D:output:0(quant_Btch_Norm-2/ReadVariableOp:value:0*quant_Btch_Norm-2/ReadVariableOp_1:value:09quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp:value:0;quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
exponential_avg_factor%fff?2$
"quant_Btch_Norm-2/FusedBatchNormV3λ
 quant_Btch_Norm-2/AssignNewValueAssignVariableOp:quant_btch_norm_2_fusedbatchnormv3_readvariableop_resource/quant_Btch_Norm-2/FusedBatchNormV3:batch_mean:02^quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp*M
_classC
A?loc:@quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02"
 quant_Btch_Norm-2/AssignNewValueω
"quant_Btch_Norm-2/AssignNewValue_1AssignVariableOp<quant_btch_norm_2_fusedbatchnormv3_readvariableop_1_resource3quant_Btch_Norm-2/FusedBatchNormV3:batch_variance:04^quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1*O
_classE
CAloc:@quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02$
"quant_Btch_Norm-2/AssignNewValue_1
quant_re_lu_7/ReluRelu&quant_Btch_Norm-2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&2
quant_re_lu_7/Relu§
%quant_re_lu_7/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%quant_re_lu_7/MovingAvgQuantize/ConstΞ
(quant_re_lu_7/MovingAvgQuantize/BatchMinMin quant_re_lu_7/Relu:activations:0.quant_re_lu_7/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2*
(quant_re_lu_7/MovingAvgQuantize/BatchMin«
'quant_re_lu_7/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2)
'quant_re_lu_7/MovingAvgQuantize/Const_1Π
(quant_re_lu_7/MovingAvgQuantize/BatchMaxMax quant_re_lu_7/Relu:activations:00quant_re_lu_7/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2*
(quant_re_lu_7/MovingAvgQuantize/BatchMax
)quant_re_lu_7/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)quant_re_lu_7/MovingAvgQuantize/Minimum/yε
'quant_re_lu_7/MovingAvgQuantize/MinimumMinimum1quant_re_lu_7/MovingAvgQuantize/BatchMin:output:02quant_re_lu_7/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2)
'quant_re_lu_7/MovingAvgQuantize/Minimum
)quant_re_lu_7/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)quant_re_lu_7/MovingAvgQuantize/Maximum/yε
'quant_re_lu_7/MovingAvgQuantize/MaximumMaximum1quant_re_lu_7/MovingAvgQuantize/BatchMax:output:02quant_re_lu_7/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2)
'quant_re_lu_7/MovingAvgQuantize/Maximum­
2quant_re_lu_7/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:24
2quant_re_lu_7/MovingAvgQuantize/AssignMinEma/decayχ
;quant_re_lu_7/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpDquant_re_lu_7_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02=
;quant_re_lu_7/MovingAvgQuantize/AssignMinEma/ReadVariableOpώ
0quant_re_lu_7/MovingAvgQuantize/AssignMinEma/subSubCquant_re_lu_7/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0+quant_re_lu_7/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 22
0quant_re_lu_7/MovingAvgQuantize/AssignMinEma/sub?
0quant_re_lu_7/MovingAvgQuantize/AssignMinEma/mulMul4quant_re_lu_7/MovingAvgQuantize/AssignMinEma/sub:z:0;quant_re_lu_7/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 22
0quant_re_lu_7/MovingAvgQuantize/AssignMinEma/mulψ
@quant_re_lu_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpDquant_re_lu_7_movingavgquantize_assignminema_readvariableop_resource4quant_re_lu_7/MovingAvgQuantize/AssignMinEma/mul:z:0<^quant_re_lu_7/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02B
@quant_re_lu_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp­
2quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:24
2quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/decayχ
;quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpDquant_re_lu_7_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02=
;quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOpώ
0quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/subSubCquant_re_lu_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0+quant_re_lu_7/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 22
0quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/sub?
0quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/mulMul4quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/sub:z:0;quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 22
0quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/mulψ
@quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpDquant_re_lu_7_movingavgquantize_assignmaxema_readvariableop_resource4quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/mul:z:0<^quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02B
@quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpΠ
Fquant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpDquant_re_lu_7_movingavgquantize_assignminema_readvariableop_resourceA^quant_re_lu_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02H
Fquant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpΤ
Hquant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpDquant_re_lu_7_movingavgquantize_assignmaxema_readvariableop_resourceA^quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02J
Hquant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
7quant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars quant_re_lu_7/Relu:activations:0Nquant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&29
7quant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVarsυ
5quant_Conv2D-3/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype027
5quant_Conv2D-3/LastValueQuant/BatchMin/ReadVariableOpΙ
8quant_Conv2D-3/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_Conv2D-3/LastValueQuant/BatchMin/reduction_indicesώ
&quant_Conv2D-3/LastValueQuant/BatchMinMin=quant_Conv2D-3/LastValueQuant/BatchMin/ReadVariableOp:value:0Aquant_Conv2D-3/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:&2(
&quant_Conv2D-3/LastValueQuant/BatchMinυ
5quant_Conv2D-3/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype027
5quant_Conv2D-3/LastValueQuant/BatchMax/ReadVariableOpΙ
8quant_Conv2D-3/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2:
8quant_Conv2D-3/LastValueQuant/BatchMax/reduction_indicesώ
&quant_Conv2D-3/LastValueQuant/BatchMaxMax=quant_Conv2D-3/LastValueQuant/BatchMax/ReadVariableOp:value:0Aquant_Conv2D-3/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:&2(
&quant_Conv2D-3/LastValueQuant/BatchMax
'quant_Conv2D-3/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2)
'quant_Conv2D-3/LastValueQuant/truediv/yα
%quant_Conv2D-3/LastValueQuant/truedivRealDiv/quant_Conv2D-3/LastValueQuant/BatchMax:output:00quant_Conv2D-3/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:&2'
%quant_Conv2D-3/LastValueQuant/truedivΪ
%quant_Conv2D-3/LastValueQuant/MinimumMinimum/quant_Conv2D-3/LastValueQuant/BatchMin:output:0)quant_Conv2D-3/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:&2'
%quant_Conv2D-3/LastValueQuant/Minimum
#quant_Conv2D-3/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2%
#quant_Conv2D-3/LastValueQuant/mul/yΡ
!quant_Conv2D-3/LastValueQuant/mulMul/quant_Conv2D-3/LastValueQuant/BatchMin:output:0,quant_Conv2D-3/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:&2#
!quant_Conv2D-3/LastValueQuant/mulΦ
%quant_Conv2D-3/LastValueQuant/MaximumMaximum/quant_Conv2D-3/LastValueQuant/BatchMax:output:0%quant_Conv2D-3/LastValueQuant/mul:z:0*
T0*
_output_shapes
:&2'
%quant_Conv2D-3/LastValueQuant/Maximumς
+quant_Conv2D-3/LastValueQuant/AssignMinLastAssignVariableOp4quant_conv2d_3_lastvaluequant_assignminlast_resource)quant_Conv2D-3/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02-
+quant_Conv2D-3/LastValueQuant/AssignMinLastς
+quant_Conv2D-3/LastValueQuant/AssignMaxLastAssignVariableOp4quant_conv2d_3_lastvaluequant_assignmaxlast_resource)quant_Conv2D-3/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02-
+quant_Conv2D-3/LastValueQuant/AssignMaxLast§
Nquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp>quant_conv2d_3_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype02P
Nquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpΓ
Pquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp4quant_conv2d_3_lastvaluequant_assignminlast_resource,^quant_Conv2D-3/LastValueQuant/AssignMinLast*
_output_shapes
:&*
dtype02R
Pquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Γ
Pquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp4quant_conv2d_3_lastvaluequant_assignmaxlast_resource,^quant_Conv2D-3/LastValueQuant/AssignMaxLast*
_output_shapes
:&*
dtype02R
Pquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ο
?quant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&&*
narrow_range(2A
?quant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel©
quant_Conv2D-3/Conv2DConv2DAquant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
quant_Conv2D-3/Conv2Dͺ
 quant_Btch_Norm-3/ReadVariableOpReadVariableOp)quant_btch_norm_3_readvariableop_resource*
_output_shapes
:&*
dtype02"
 quant_Btch_Norm-3/ReadVariableOp°
"quant_Btch_Norm-3/ReadVariableOp_1ReadVariableOp+quant_btch_norm_3_readvariableop_1_resource*
_output_shapes
:&*
dtype02$
"quant_Btch_Norm-3/ReadVariableOp_1έ
1quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOpReadVariableOp:quant_btch_norm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype023
1quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOpγ
3quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<quant_btch_norm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype025
3quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1ά
"quant_Btch_Norm-3/FusedBatchNormV3FusedBatchNormV3quant_Conv2D-3/Conv2D:output:0(quant_Btch_Norm-3/ReadVariableOp:value:0*quant_Btch_Norm-3/ReadVariableOp_1:value:09quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp:value:0;quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
exponential_avg_factor%fff?2$
"quant_Btch_Norm-3/FusedBatchNormV3λ
 quant_Btch_Norm-3/AssignNewValueAssignVariableOp:quant_btch_norm_3_fusedbatchnormv3_readvariableop_resource/quant_Btch_Norm-3/FusedBatchNormV3:batch_mean:02^quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp*M
_classC
A?loc:@quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02"
 quant_Btch_Norm-3/AssignNewValueω
"quant_Btch_Norm-3/AssignNewValue_1AssignVariableOp<quant_btch_norm_3_fusedbatchnormv3_readvariableop_1_resource3quant_Btch_Norm-3/FusedBatchNormV3:batch_variance:04^quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1*O
_classE
CAloc:@quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02$
"quant_Btch_Norm-3/AssignNewValue_1
quant_re_lu_8/ReluRelu&quant_Btch_Norm-3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&2
quant_re_lu_8/Relu§
%quant_re_lu_8/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2'
%quant_re_lu_8/MovingAvgQuantize/ConstΞ
(quant_re_lu_8/MovingAvgQuantize/BatchMinMin quant_re_lu_8/Relu:activations:0.quant_re_lu_8/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2*
(quant_re_lu_8/MovingAvgQuantize/BatchMin«
'quant_re_lu_8/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2)
'quant_re_lu_8/MovingAvgQuantize/Const_1Π
(quant_re_lu_8/MovingAvgQuantize/BatchMaxMax quant_re_lu_8/Relu:activations:00quant_re_lu_8/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2*
(quant_re_lu_8/MovingAvgQuantize/BatchMax
)quant_re_lu_8/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)quant_re_lu_8/MovingAvgQuantize/Minimum/yε
'quant_re_lu_8/MovingAvgQuantize/MinimumMinimum1quant_re_lu_8/MovingAvgQuantize/BatchMin:output:02quant_re_lu_8/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2)
'quant_re_lu_8/MovingAvgQuantize/Minimum
)quant_re_lu_8/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)quant_re_lu_8/MovingAvgQuantize/Maximum/yε
'quant_re_lu_8/MovingAvgQuantize/MaximumMaximum1quant_re_lu_8/MovingAvgQuantize/BatchMax:output:02quant_re_lu_8/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2)
'quant_re_lu_8/MovingAvgQuantize/Maximum­
2quant_re_lu_8/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:24
2quant_re_lu_8/MovingAvgQuantize/AssignMinEma/decayχ
;quant_re_lu_8/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpDquant_re_lu_8_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02=
;quant_re_lu_8/MovingAvgQuantize/AssignMinEma/ReadVariableOpώ
0quant_re_lu_8/MovingAvgQuantize/AssignMinEma/subSubCquant_re_lu_8/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0+quant_re_lu_8/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 22
0quant_re_lu_8/MovingAvgQuantize/AssignMinEma/sub?
0quant_re_lu_8/MovingAvgQuantize/AssignMinEma/mulMul4quant_re_lu_8/MovingAvgQuantize/AssignMinEma/sub:z:0;quant_re_lu_8/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 22
0quant_re_lu_8/MovingAvgQuantize/AssignMinEma/mulψ
@quant_re_lu_8/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpDquant_re_lu_8_movingavgquantize_assignminema_readvariableop_resource4quant_re_lu_8/MovingAvgQuantize/AssignMinEma/mul:z:0<^quant_re_lu_8/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02B
@quant_re_lu_8/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp­
2quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:24
2quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/decayχ
;quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpDquant_re_lu_8_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02=
;quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/ReadVariableOpώ
0quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/subSubCquant_re_lu_8/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0+quant_re_lu_8/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 22
0quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/sub?
0quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/mulMul4quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/sub:z:0;quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 22
0quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/mulψ
@quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpDquant_re_lu_8_movingavgquantize_assignmaxema_readvariableop_resource4quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/mul:z:0<^quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02B
@quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpΠ
Fquant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpDquant_re_lu_8_movingavgquantize_assignminema_readvariableop_resourceA^quant_re_lu_8/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02H
Fquant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpΤ
Hquant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpDquant_re_lu_8_movingavgquantize_assignmaxema_readvariableop_resourceA^quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02J
Hquant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
7quant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars quant_re_lu_8/Relu:activations:0Nquant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&29
7quant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVarsΓ
7quant_GlobalAveragePooling-Layer/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7quant_GlobalAveragePooling-Layer/Mean/reduction_indices
%quant_GlobalAveragePooling-Layer/MeanMeanAquant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0@quant_GlobalAveragePooling-Layer/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????&2'
%quant_GlobalAveragePooling-Layer/MeanΕ
8quant_GlobalAveragePooling-Layer/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2:
8quant_GlobalAveragePooling-Layer/MovingAvgQuantize/Const
;quant_GlobalAveragePooling-Layer/MovingAvgQuantize/BatchMinMin.quant_GlobalAveragePooling-Layer/Mean:output:0Aquant_GlobalAveragePooling-Layer/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2=
;quant_GlobalAveragePooling-Layer/MovingAvgQuantize/BatchMinΙ
:quant_GlobalAveragePooling-Layer/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2<
:quant_GlobalAveragePooling-Layer/MovingAvgQuantize/Const_1
;quant_GlobalAveragePooling-Layer/MovingAvgQuantize/BatchMaxMax.quant_GlobalAveragePooling-Layer/Mean:output:0Cquant_GlobalAveragePooling-Layer/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2=
;quant_GlobalAveragePooling-Layer/MovingAvgQuantize/BatchMaxΑ
<quant_GlobalAveragePooling-Layer/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2>
<quant_GlobalAveragePooling-Layer/MovingAvgQuantize/Minimum/y±
:quant_GlobalAveragePooling-Layer/MovingAvgQuantize/MinimumMinimumDquant_GlobalAveragePooling-Layer/MovingAvgQuantize/BatchMin:output:0Equant_GlobalAveragePooling-Layer/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2<
:quant_GlobalAveragePooling-Layer/MovingAvgQuantize/MinimumΑ
<quant_GlobalAveragePooling-Layer/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2>
<quant_GlobalAveragePooling-Layer/MovingAvgQuantize/Maximum/y±
:quant_GlobalAveragePooling-Layer/MovingAvgQuantize/MaximumMaximumDquant_GlobalAveragePooling-Layer/MovingAvgQuantize/BatchMax:output:0Equant_GlobalAveragePooling-Layer/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2<
:quant_GlobalAveragePooling-Layer/MovingAvgQuantize/MaximumΣ
Equant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2G
Equant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/decay°
Nquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpWquant_globalaveragepooling_layer_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02P
Nquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/ReadVariableOpΚ
Cquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/subSubVquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0>quant_GlobalAveragePooling-Layer/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2E
Cquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/subΛ
Cquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/mulMulGquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/sub:z:0Nquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2E
Cquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/mulΧ
Squant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpWquant_globalaveragepooling_layer_movingavgquantize_assignminema_readvariableop_resourceGquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/mul:z:0O^quant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02U
Squant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpΣ
Equant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2G
Equant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/decay°
Nquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpWquant_globalaveragepooling_layer_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02P
Nquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/ReadVariableOpΚ
Cquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/subSubVquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0>quant_GlobalAveragePooling-Layer/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2E
Cquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/subΛ
Cquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/mulMulGquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/sub:z:0Nquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2E
Cquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/mulΧ
Squant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpWquant_globalaveragepooling_layer_movingavgquantize_assignmaxema_readvariableop_resourceGquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/mul:z:0O^quant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02U
Squant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
Yquant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpWquant_globalaveragepooling_layer_movingavgquantize_assignminema_readvariableop_resourceT^quant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02[
Yquant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp 
[quant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpWquant_globalaveragepooling_layer_movingavgquantize_assignmaxema_readvariableop_resourceT^quant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02]
[quant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Τ
Jquant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars.quant_GlobalAveragePooling-Layer/Mean:output:0aquant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0cquant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????&2L
Jquant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVarsν
5quant_Output-Layer/LastValueQuant/Rank/ReadVariableOpReadVariableOp>quant_output_layer_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:&*
dtype027
5quant_Output-Layer/LastValueQuant/Rank/ReadVariableOp
&quant_Output-Layer/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :2(
&quant_Output-Layer/LastValueQuant/Rank 
-quant_Output-Layer/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2/
-quant_Output-Layer/LastValueQuant/range/start 
-quant_Output-Layer/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2/
-quant_Output-Layer/LastValueQuant/range/delta
'quant_Output-Layer/LastValueQuant/rangeRange6quant_Output-Layer/LastValueQuant/range/start:output:0/quant_Output-Layer/LastValueQuant/Rank:output:06quant_Output-Layer/LastValueQuant/range/delta:output:0*
_output_shapes
:2)
'quant_Output-Layer/LastValueQuant/rangeυ
9quant_Output-Layer/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp>quant_output_layer_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:&*
dtype02;
9quant_Output-Layer/LastValueQuant/BatchMin/ReadVariableOpυ
*quant_Output-Layer/LastValueQuant/BatchMinMinAquant_Output-Layer/LastValueQuant/BatchMin/ReadVariableOp:value:00quant_Output-Layer/LastValueQuant/range:output:0*
T0*
_output_shapes
: 2,
*quant_Output-Layer/LastValueQuant/BatchMinρ
7quant_Output-Layer/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp>quant_output_layer_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:&*
dtype029
7quant_Output-Layer/LastValueQuant/Rank_1/ReadVariableOp
(quant_Output-Layer/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2*
(quant_Output-Layer/LastValueQuant/Rank_1€
/quant_Output-Layer/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 21
/quant_Output-Layer/LastValueQuant/range_1/start€
/quant_Output-Layer/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :21
/quant_Output-Layer/LastValueQuant/range_1/delta’
)quant_Output-Layer/LastValueQuant/range_1Range8quant_Output-Layer/LastValueQuant/range_1/start:output:01quant_Output-Layer/LastValueQuant/Rank_1:output:08quant_Output-Layer/LastValueQuant/range_1/delta:output:0*
_output_shapes
:2+
)quant_Output-Layer/LastValueQuant/range_1υ
9quant_Output-Layer/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp>quant_output_layer_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:&*
dtype02;
9quant_Output-Layer/LastValueQuant/BatchMax/ReadVariableOpχ
*quant_Output-Layer/LastValueQuant/BatchMaxMaxAquant_Output-Layer/LastValueQuant/BatchMax/ReadVariableOp:value:02quant_Output-Layer/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: 2,
*quant_Output-Layer/LastValueQuant/BatchMax
+quant_Output-Layer/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2-
+quant_Output-Layer/LastValueQuant/truediv/yν
)quant_Output-Layer/LastValueQuant/truedivRealDiv3quant_Output-Layer/LastValueQuant/BatchMax:output:04quant_Output-Layer/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 2+
)quant_Output-Layer/LastValueQuant/truedivζ
)quant_Output-Layer/LastValueQuant/MinimumMinimum3quant_Output-Layer/LastValueQuant/BatchMin:output:0-quant_Output-Layer/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: 2+
)quant_Output-Layer/LastValueQuant/Minimum
'quant_Output-Layer/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2)
'quant_Output-Layer/LastValueQuant/mul/yέ
%quant_Output-Layer/LastValueQuant/mulMul3quant_Output-Layer/LastValueQuant/BatchMin:output:00quant_Output-Layer/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: 2'
%quant_Output-Layer/LastValueQuant/mulβ
)quant_Output-Layer/LastValueQuant/MaximumMaximum3quant_Output-Layer/LastValueQuant/BatchMax:output:0)quant_Output-Layer/LastValueQuant/mul:z:0*
T0*
_output_shapes
: 2+
)quant_Output-Layer/LastValueQuant/Maximum
/quant_Output-Layer/LastValueQuant/AssignMinLastAssignVariableOp8quant_output_layer_lastvaluequant_assignminlast_resource-quant_Output-Layer/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype021
/quant_Output-Layer/LastValueQuant/AssignMinLast
/quant_Output-Layer/LastValueQuant/AssignMaxLastAssignVariableOp8quant_output_layer_lastvaluequant_assignmaxlast_resource-quant_Output-Layer/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype021
/quant_Output-Layer/LastValueQuant/AssignMaxLast
Hquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>quant_output_layer_lastvaluequant_rank_readvariableop_resource*
_output_shapes

:&*
dtype02J
Hquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp»
Jquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp8quant_output_layer_lastvaluequant_assignminlast_resource0^quant_Output-Layer/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype02L
Jquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1»
Jquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp8quant_output_layer_lastvaluequant_assignmaxlast_resource0^quant_Output-Layer/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype02L
Jquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Ώ
9quant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsPquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Rquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Rquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:&*
narrow_range(2;
9quant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars
quant_Output-Layer/MatMulMatMulTquant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Cquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2
quant_Output-Layer/MatMulΕ
)quant_Output-Layer/BiasAdd/ReadVariableOpReadVariableOp2quant_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)quant_Output-Layer/BiasAdd/ReadVariableOpΝ
quant_Output-Layer/BiasAddBiasAdd#quant_Output-Layer/MatMul:product:01quant_Output-Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
quant_Output-Layer/BiasAdd©
*quant_Output-Layer/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2,
*quant_Output-Layer/MovingAvgQuantize/Constΰ
-quant_Output-Layer/MovingAvgQuantize/BatchMinMin#quant_Output-Layer/BiasAdd:output:03quant_Output-Layer/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2/
-quant_Output-Layer/MovingAvgQuantize/BatchMin­
,quant_Output-Layer/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2.
,quant_Output-Layer/MovingAvgQuantize/Const_1β
-quant_Output-Layer/MovingAvgQuantize/BatchMaxMax#quant_Output-Layer/BiasAdd:output:05quant_Output-Layer/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2/
-quant_Output-Layer/MovingAvgQuantize/BatchMax₯
.quant_Output-Layer/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.quant_Output-Layer/MovingAvgQuantize/Minimum/yω
,quant_Output-Layer/MovingAvgQuantize/MinimumMinimum6quant_Output-Layer/MovingAvgQuantize/BatchMin:output:07quant_Output-Layer/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2.
,quant_Output-Layer/MovingAvgQuantize/Minimum₯
.quant_Output-Layer/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.quant_Output-Layer/MovingAvgQuantize/Maximum/yω
,quant_Output-Layer/MovingAvgQuantize/MaximumMaximum6quant_Output-Layer/MovingAvgQuantize/BatchMax:output:07quant_Output-Layer/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2.
,quant_Output-Layer/MovingAvgQuantize/Maximum·
7quant_Output-Layer/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:29
7quant_Output-Layer/MovingAvgQuantize/AssignMinEma/decay
@quant_Output-Layer/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpIquant_output_layer_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02B
@quant_Output-Layer/MovingAvgQuantize/AssignMinEma/ReadVariableOp
5quant_Output-Layer/MovingAvgQuantize/AssignMinEma/subSubHquant_Output-Layer/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:00quant_Output-Layer/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 27
5quant_Output-Layer/MovingAvgQuantize/AssignMinEma/sub
5quant_Output-Layer/MovingAvgQuantize/AssignMinEma/mulMul9quant_Output-Layer/MovingAvgQuantize/AssignMinEma/sub:z:0@quant_Output-Layer/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 27
5quant_Output-Layer/MovingAvgQuantize/AssignMinEma/mul
Equant_Output-Layer/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpIquant_output_layer_movingavgquantize_assignminema_readvariableop_resource9quant_Output-Layer/MovingAvgQuantize/AssignMinEma/mul:z:0A^quant_Output-Layer/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02G
Equant_Output-Layer/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp·
7quant_Output-Layer/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:29
7quant_Output-Layer/MovingAvgQuantize/AssignMaxEma/decay
@quant_Output-Layer/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpIquant_output_layer_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02B
@quant_Output-Layer/MovingAvgQuantize/AssignMaxEma/ReadVariableOp
5quant_Output-Layer/MovingAvgQuantize/AssignMaxEma/subSubHquant_Output-Layer/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:00quant_Output-Layer/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 27
5quant_Output-Layer/MovingAvgQuantize/AssignMaxEma/sub
5quant_Output-Layer/MovingAvgQuantize/AssignMaxEma/mulMul9quant_Output-Layer/MovingAvgQuantize/AssignMaxEma/sub:z:0@quant_Output-Layer/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 27
5quant_Output-Layer/MovingAvgQuantize/AssignMaxEma/mul
Equant_Output-Layer/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpIquant_output_layer_movingavgquantize_assignmaxema_readvariableop_resource9quant_Output-Layer/MovingAvgQuantize/AssignMaxEma/mul:z:0A^quant_Output-Layer/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02G
Equant_Output-Layer/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpδ
Kquant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpIquant_output_layer_movingavgquantize_assignminema_readvariableop_resourceF^quant_Output-Layer/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02M
Kquant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpθ
Mquant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpIquant_output_layer_movingavgquantize_assignmaxema_readvariableop_resourceF^quant_Output-Layer/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02O
Mquant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
<quant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars#quant_Output-Layer/BiasAdd:output:0Squant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Uquant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????2>
<quant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars’
IdentityIdentityFquant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0!^quant_Btch_Norm-1/AssignNewValue#^quant_Btch_Norm-1/AssignNewValue_1!^quant_Btch_Norm-2/AssignNewValue#^quant_Btch_Norm-2/AssignNewValue_1!^quant_Btch_Norm-3/AssignNewValue#^quant_Btch_Norm-3/AssignNewValue_1,^quant_Conv2D-1/LastValueQuant/AssignMaxLast,^quant_Conv2D-1/LastValueQuant/AssignMinLast,^quant_Conv2D-2/LastValueQuant/AssignMaxLast,^quant_Conv2D-2/LastValueQuant/AssignMinLast,^quant_Conv2D-3/LastValueQuant/AssignMaxLast,^quant_Conv2D-3/LastValueQuant/AssignMinLastT^quant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpT^quant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp0^quant_Output-Layer/LastValueQuant/AssignMaxLast0^quant_Output-Layer/LastValueQuant/AssignMinLastF^quant_Output-Layer/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpF^quant_Output-Layer/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpA^quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpA^quant_re_lu_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpA^quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpA^quant_re_lu_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpA^quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpA^quant_re_lu_8/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp3^quantize_layer/AllValuesQuantize/AssignMaxAllValue3^quantize_layer/AllValuesQuantize/AssignMinAllValue*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Δ
_input_shapes²
―:?????????1
:::::::::::::::::::::::::::::::::::::2D
 quant_Btch_Norm-1/AssignNewValue quant_Btch_Norm-1/AssignNewValue2H
"quant_Btch_Norm-1/AssignNewValue_1"quant_Btch_Norm-1/AssignNewValue_12D
 quant_Btch_Norm-2/AssignNewValue quant_Btch_Norm-2/AssignNewValue2H
"quant_Btch_Norm-2/AssignNewValue_1"quant_Btch_Norm-2/AssignNewValue_12D
 quant_Btch_Norm-3/AssignNewValue quant_Btch_Norm-3/AssignNewValue2H
"quant_Btch_Norm-3/AssignNewValue_1"quant_Btch_Norm-3/AssignNewValue_12Z
+quant_Conv2D-1/LastValueQuant/AssignMaxLast+quant_Conv2D-1/LastValueQuant/AssignMaxLast2Z
+quant_Conv2D-1/LastValueQuant/AssignMinLast+quant_Conv2D-1/LastValueQuant/AssignMinLast2Z
+quant_Conv2D-2/LastValueQuant/AssignMaxLast+quant_Conv2D-2/LastValueQuant/AssignMaxLast2Z
+quant_Conv2D-2/LastValueQuant/AssignMinLast+quant_Conv2D-2/LastValueQuant/AssignMinLast2Z
+quant_Conv2D-3/LastValueQuant/AssignMaxLast+quant_Conv2D-3/LastValueQuant/AssignMaxLast2Z
+quant_Conv2D-3/LastValueQuant/AssignMinLast+quant_Conv2D-3/LastValueQuant/AssignMinLast2ͺ
Squant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpSquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2ͺ
Squant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpSquant_GlobalAveragePooling-Layer/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2b
/quant_Output-Layer/LastValueQuant/AssignMaxLast/quant_Output-Layer/LastValueQuant/AssignMaxLast2b
/quant_Output-Layer/LastValueQuant/AssignMinLast/quant_Output-Layer/LastValueQuant/AssignMinLast2
Equant_Output-Layer/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpEquant_Output-Layer/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2
Equant_Output-Layer/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpEquant_Output-Layer/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2
@quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp@quant_re_lu_6/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2
@quant_re_lu_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp@quant_re_lu_6/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2
@quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp@quant_re_lu_7/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2
@quant_re_lu_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp@quant_re_lu_7/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2
@quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp@quant_re_lu_8/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2
@quant_re_lu_8/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp@quant_re_lu_8/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2h
2quantize_layer/AllValuesQuantize/AssignMaxAllValue2quantize_layer/AllValuesQuantize/AssignMaxAllValue2h
2quantize_layer/AllValuesQuantize/AssignMinAllValue2quantize_layer/AllValuesQuantize/AssignMinAllValue:W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs

Φ
I__inference_quant_Conv2D-2_layer_call_and_return_conditional_losses_59572

inputsL
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resourceN
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resourceN
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource
identity
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:&&*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2€
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&&*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelΑ
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
Conv2Dk
IdentityIdentityConv2D:output:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????&::::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs


-__inference_quant_re_lu_7_layer_call_fn_59712

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_7_layer_call_and_return_conditional_losses_578332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
Λ

L__inference_quant_Btch_Norm-2_layer_call_and_return_conditional_losses_57760

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&:::::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
ΛM

G__inference_sequential_4_layer_call_and_return_conditional_losses_58250
conv2d_1_input
quantize_layer_57447
quantize_layer_57449
quant_conv2d_1_57512
quant_conv2d_1_57514
quant_conv2d_1_57516
quant_btch_norm_1_57585
quant_btch_norm_1_57587
quant_btch_norm_1_57589
quant_btch_norm_1_57591
quant_re_lu_6_57650
quant_re_lu_6_57652
quant_conv2d_2_57715
quant_conv2d_2_57717
quant_conv2d_2_57719
quant_btch_norm_2_57788
quant_btch_norm_2_57790
quant_btch_norm_2_57792
quant_btch_norm_2_57794
quant_re_lu_7_57853
quant_re_lu_7_57855
quant_conv2d_3_57918
quant_conv2d_3_57920
quant_conv2d_3_57922
quant_btch_norm_3_57991
quant_btch_norm_3_57993
quant_btch_norm_3_57995
quant_btch_norm_3_57997
quant_re_lu_8_58056
quant_re_lu_8_58058*
&quant_globalaveragepooling_layer_58119*
&quant_globalaveragepooling_layer_58121
quant_output_layer_58236
quant_output_layer_58238
quant_output_layer_58240
quant_output_layer_58242
quant_output_layer_58244
quant_output_layer_58246
identity’)quant_Btch_Norm-1/StatefulPartitionedCall’)quant_Btch_Norm-2/StatefulPartitionedCall’)quant_Btch_Norm-3/StatefulPartitionedCall’&quant_Conv2D-1/StatefulPartitionedCall’&quant_Conv2D-2/StatefulPartitionedCall’&quant_Conv2D-3/StatefulPartitionedCall’8quant_GlobalAveragePooling-Layer/StatefulPartitionedCall’*quant_Output-Layer/StatefulPartitionedCall’%quant_re_lu_6/StatefulPartitionedCall’%quant_re_lu_7/StatefulPartitionedCall’%quant_re_lu_8/StatefulPartitionedCall’&quantize_layer/StatefulPartitionedCall»
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputquantize_layer_57447quantize_layer_57449*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quantize_layer_layer_call_and_return_conditional_losses_574182(
&quantize_layer/StatefulPartitionedCallφ
&quant_Conv2D-1/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_1_57512quant_conv2d_1_57514quant_conv2d_1_57516*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-1_layer_call_and_return_conditional_losses_574762(
&quant_Conv2D-1/StatefulPartitionedCall£
)quant_Btch_Norm-1/StatefulPartitionedCallStatefulPartitionedCall/quant_Conv2D-1/StatefulPartitionedCall:output:0quant_btch_norm_1_57585quant_btch_norm_1_57587quant_btch_norm_1_57589quant_btch_norm_1_57591*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-1_layer_call_and_return_conditional_losses_575392+
)quant_Btch_Norm-1/StatefulPartitionedCallΪ
%quant_re_lu_6/StatefulPartitionedCallStatefulPartitionedCall2quant_Btch_Norm-1/StatefulPartitionedCall:output:0quant_re_lu_6_57650quant_re_lu_6_57652*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_6_layer_call_and_return_conditional_losses_576202'
%quant_re_lu_6/StatefulPartitionedCallυ
&quant_Conv2D-2/StatefulPartitionedCallStatefulPartitionedCall.quant_re_lu_6/StatefulPartitionedCall:output:0quant_conv2d_2_57715quant_conv2d_2_57717quant_conv2d_2_57719*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-2_layer_call_and_return_conditional_losses_576792(
&quant_Conv2D-2/StatefulPartitionedCall£
)quant_Btch_Norm-2/StatefulPartitionedCallStatefulPartitionedCall/quant_Conv2D-2/StatefulPartitionedCall:output:0quant_btch_norm_2_57788quant_btch_norm_2_57790quant_btch_norm_2_57792quant_btch_norm_2_57794*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-2_layer_call_and_return_conditional_losses_577422+
)quant_Btch_Norm-2/StatefulPartitionedCallΪ
%quant_re_lu_7/StatefulPartitionedCallStatefulPartitionedCall2quant_Btch_Norm-2/StatefulPartitionedCall:output:0quant_re_lu_7_57853quant_re_lu_7_57855*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_7_layer_call_and_return_conditional_losses_578232'
%quant_re_lu_7/StatefulPartitionedCallυ
&quant_Conv2D-3/StatefulPartitionedCallStatefulPartitionedCall.quant_re_lu_7/StatefulPartitionedCall:output:0quant_conv2d_3_57918quant_conv2d_3_57920quant_conv2d_3_57922*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-3_layer_call_and_return_conditional_losses_578822(
&quant_Conv2D-3/StatefulPartitionedCall£
)quant_Btch_Norm-3/StatefulPartitionedCallStatefulPartitionedCall/quant_Conv2D-3/StatefulPartitionedCall:output:0quant_btch_norm_3_57991quant_btch_norm_3_57993quant_btch_norm_3_57995quant_btch_norm_3_57997*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-3_layer_call_and_return_conditional_losses_579452+
)quant_Btch_Norm-3/StatefulPartitionedCallΪ
%quant_re_lu_8/StatefulPartitionedCallStatefulPartitionedCall2quant_Btch_Norm-3/StatefulPartitionedCall:output:0quant_re_lu_8_58056quant_re_lu_8_58058*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_8_layer_call_and_return_conditional_losses_580262'
%quant_re_lu_8/StatefulPartitionedCall­
8quant_GlobalAveragePooling-Layer/StatefulPartitionedCallStatefulPartitionedCall.quant_re_lu_8/StatefulPartitionedCall:output:0&quant_globalaveragepooling_layer_58119&quant_globalaveragepooling_layer_58121*
Tin
2*
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
GPU 2J 8 *d
f_R]
[__inference_quant_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_580882:
8quant_GlobalAveragePooling-Layer/StatefulPartitionedCallκ
*quant_Output-Layer/StatefulPartitionedCallStatefulPartitionedCallAquant_GlobalAveragePooling-Layer/StatefulPartitionedCall:output:0quant_output_layer_58236quant_output_layer_58238quant_output_layer_58240quant_output_layer_58242quant_output_layer_58244quant_output_layer_58246*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_quant_Output-Layer_layer_call_and_return_conditional_losses_581802,
*quant_Output-Layer/StatefulPartitionedCall
IdentityIdentity3quant_Output-Layer/StatefulPartitionedCall:output:0*^quant_Btch_Norm-1/StatefulPartitionedCall*^quant_Btch_Norm-2/StatefulPartitionedCall*^quant_Btch_Norm-3/StatefulPartitionedCall'^quant_Conv2D-1/StatefulPartitionedCall'^quant_Conv2D-2/StatefulPartitionedCall'^quant_Conv2D-3/StatefulPartitionedCall9^quant_GlobalAveragePooling-Layer/StatefulPartitionedCall+^quant_Output-Layer/StatefulPartitionedCall&^quant_re_lu_6/StatefulPartitionedCall&^quant_re_lu_7/StatefulPartitionedCall&^quant_re_lu_8/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Δ
_input_shapes²
―:?????????1
:::::::::::::::::::::::::::::::::::::2V
)quant_Btch_Norm-1/StatefulPartitionedCall)quant_Btch_Norm-1/StatefulPartitionedCall2V
)quant_Btch_Norm-2/StatefulPartitionedCall)quant_Btch_Norm-2/StatefulPartitionedCall2V
)quant_Btch_Norm-3/StatefulPartitionedCall)quant_Btch_Norm-3/StatefulPartitionedCall2P
&quant_Conv2D-1/StatefulPartitionedCall&quant_Conv2D-1/StatefulPartitionedCall2P
&quant_Conv2D-2/StatefulPartitionedCall&quant_Conv2D-2/StatefulPartitionedCall2P
&quant_Conv2D-3/StatefulPartitionedCall&quant_Conv2D-3/StatefulPartitionedCall2t
8quant_GlobalAveragePooling-Layer/StatefulPartitionedCall8quant_GlobalAveragePooling-Layer/StatefulPartitionedCall2X
*quant_Output-Layer/StatefulPartitionedCall*quant_Output-Layer/StatefulPartitionedCall2N
%quant_re_lu_6/StatefulPartitionedCall%quant_re_lu_6/StatefulPartitionedCall2N
%quant_re_lu_7/StatefulPartitionedCall%quant_re_lu_7/StatefulPartitionedCall2N
%quant_re_lu_8/StatefulPartitionedCall%quant_re_lu_8/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:_ [
/
_output_shapes
:?????????1

(
_user_specified_nameConv2D-1_input

$
__inference__traced_save_60509
file_prefix@
<savev2_quantize_layer_quantize_layer_min_read_readvariableop@
<savev2_quantize_layer_quantize_layer_max_read_readvariableop<
8savev2_quantize_layer_optimizer_step_read_readvariableop<
8savev2_quant_conv2d_1_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_1_kernel_min_read_readvariableop8
4savev2_quant_conv2d_1_kernel_max_read_readvariableop?
;savev2_quant_btch_norm_1_optimizer_step_read_readvariableop;
7savev2_quant_re_lu_6_optimizer_step_read_readvariableop7
3savev2_quant_re_lu_6_output_min_read_readvariableop7
3savev2_quant_re_lu_6_output_max_read_readvariableop<
8savev2_quant_conv2d_2_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_2_kernel_min_read_readvariableop8
4savev2_quant_conv2d_2_kernel_max_read_readvariableop?
;savev2_quant_btch_norm_2_optimizer_step_read_readvariableop;
7savev2_quant_re_lu_7_optimizer_step_read_readvariableop7
3savev2_quant_re_lu_7_output_min_read_readvariableop7
3savev2_quant_re_lu_7_output_max_read_readvariableop<
8savev2_quant_conv2d_3_optimizer_step_read_readvariableop8
4savev2_quant_conv2d_3_kernel_min_read_readvariableop8
4savev2_quant_conv2d_3_kernel_max_read_readvariableop?
;savev2_quant_btch_norm_3_optimizer_step_read_readvariableop;
7savev2_quant_re_lu_8_optimizer_step_read_readvariableop7
3savev2_quant_re_lu_8_output_min_read_readvariableop7
3savev2_quant_re_lu_8_output_max_read_readvariableopN
Jsavev2_quant_globalaveragepooling_layer_optimizer_step_read_readvariableopJ
Fsavev2_quant_globalaveragepooling_layer_output_min_read_readvariableopJ
Fsavev2_quant_globalaveragepooling_layer_output_max_read_readvariableop@
<savev2_quant_output_layer_optimizer_step_read_readvariableop<
8savev2_quant_output_layer_kernel_min_read_readvariableop<
8savev2_quant_output_layer_kernel_max_read_readvariableopE
Asavev2_quant_output_layer_post_activation_min_read_readvariableopE
Asavev2_quant_output_layer_post_activation_max_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
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
6savev2_btch_norm_3_moving_variance_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop$
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
2savev2_adam_btch_norm_3_beta_m_read_readvariableop7
3savev2_adam_output_layer_bias_m_read_readvariableop9
5savev2_adam_output_layer_kernel_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop7
3savev2_adam_btch_norm_1_gamma_v_read_readvariableop6
2savev2_adam_btch_norm_1_beta_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop7
3savev2_adam_btch_norm_2_gamma_v_read_readvariableop6
2savev2_adam_btch_norm_2_beta_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop7
3savev2_adam_btch_norm_3_gamma_v_read_readvariableop6
2savev2_adam_btch_norm_3_beta_v_read_readvariableop7
3savev2_adam_output_layer_bias_v_read_readvariableop9
5savev2_adam_output_layer_kernel_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_20c96e9eaaf74ad9942028b1727b8bf5/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameΥ%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*η$
valueέ$BΪ$QBBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/output_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/output_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/kernel_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/output_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-10/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-10/output_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-10/output_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-11/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-11/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-11/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-11/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-11/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names­
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*·
value­BͺQB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesυ"
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0<savev2_quantize_layer_quantize_layer_min_read_readvariableop<savev2_quantize_layer_quantize_layer_max_read_readvariableop8savev2_quantize_layer_optimizer_step_read_readvariableop8savev2_quant_conv2d_1_optimizer_step_read_readvariableop4savev2_quant_conv2d_1_kernel_min_read_readvariableop4savev2_quant_conv2d_1_kernel_max_read_readvariableop;savev2_quant_btch_norm_1_optimizer_step_read_readvariableop7savev2_quant_re_lu_6_optimizer_step_read_readvariableop3savev2_quant_re_lu_6_output_min_read_readvariableop3savev2_quant_re_lu_6_output_max_read_readvariableop8savev2_quant_conv2d_2_optimizer_step_read_readvariableop4savev2_quant_conv2d_2_kernel_min_read_readvariableop4savev2_quant_conv2d_2_kernel_max_read_readvariableop;savev2_quant_btch_norm_2_optimizer_step_read_readvariableop7savev2_quant_re_lu_7_optimizer_step_read_readvariableop3savev2_quant_re_lu_7_output_min_read_readvariableop3savev2_quant_re_lu_7_output_max_read_readvariableop8savev2_quant_conv2d_3_optimizer_step_read_readvariableop4savev2_quant_conv2d_3_kernel_min_read_readvariableop4savev2_quant_conv2d_3_kernel_max_read_readvariableop;savev2_quant_btch_norm_3_optimizer_step_read_readvariableop7savev2_quant_re_lu_8_optimizer_step_read_readvariableop3savev2_quant_re_lu_8_output_min_read_readvariableop3savev2_quant_re_lu_8_output_max_read_readvariableopJsavev2_quant_globalaveragepooling_layer_optimizer_step_read_readvariableopFsavev2_quant_globalaveragepooling_layer_output_min_read_readvariableopFsavev2_quant_globalaveragepooling_layer_output_max_read_readvariableop<savev2_quant_output_layer_optimizer_step_read_readvariableop8savev2_quant_output_layer_kernel_min_read_readvariableop8savev2_quant_output_layer_kernel_max_read_readvariableopAsavev2_quant_output_layer_post_activation_min_read_readvariableopAsavev2_quant_output_layer_post_activation_max_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop,savev2_btch_norm_1_gamma_read_readvariableop+savev2_btch_norm_1_beta_read_readvariableop2savev2_btch_norm_1_moving_mean_read_readvariableop6savev2_btch_norm_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop,savev2_btch_norm_2_gamma_read_readvariableop+savev2_btch_norm_2_beta_read_readvariableop2savev2_btch_norm_2_moving_mean_read_readvariableop6savev2_btch_norm_2_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop,savev2_btch_norm_3_gamma_read_readvariableop+savev2_btch_norm_3_beta_read_readvariableop2savev2_btch_norm_3_moving_mean_read_readvariableop6savev2_btch_norm_3_moving_variance_read_readvariableop,savev2_output_layer_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop3savev2_adam_btch_norm_1_gamma_m_read_readvariableop2savev2_adam_btch_norm_1_beta_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop3savev2_adam_btch_norm_2_gamma_m_read_readvariableop2savev2_adam_btch_norm_2_beta_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop3savev2_adam_btch_norm_3_gamma_m_read_readvariableop2savev2_adam_btch_norm_3_beta_m_read_readvariableop3savev2_adam_output_layer_bias_m_read_readvariableop5savev2_adam_output_layer_kernel_m_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop3savev2_adam_btch_norm_1_gamma_v_read_readvariableop2savev2_adam_btch_norm_1_beta_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop3savev2_adam_btch_norm_2_gamma_v_read_readvariableop2savev2_adam_btch_norm_2_beta_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop3savev2_adam_btch_norm_3_gamma_v_read_readvariableop2savev2_adam_btch_norm_3_beta_v_read_readvariableop3savev2_adam_output_layer_bias_v_read_readvariableop5savev2_adam_output_layer_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *_
dtypesU
S2Q	2
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ε
_input_shapesΣ
Π: : : : : :&:&: : : : : :&:&: : : : : :&:&: : : : : : : : : : : : : : : : : :&:&:&:&:&:&&:&:&:&:&:&&:&:&:&:&::&: : : : :&:&:&:&&:&:&:&&:&:&::&:&:&:&:&&:&:&:&&:&:&::&: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:&: 

_output_shapes
:&:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: : 

_output_shapes
:&: 

_output_shapes
:&:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:&: 

_output_shapes
:&:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :,&(
&
_output_shapes
:&: '

_output_shapes
:&: (

_output_shapes
:&: )

_output_shapes
:&: *

_output_shapes
:&:,+(
&
_output_shapes
:&&: ,

_output_shapes
:&: -

_output_shapes
:&: .

_output_shapes
:&: /

_output_shapes
:&:,0(
&
_output_shapes
:&&: 1

_output_shapes
:&: 2

_output_shapes
:&: 3

_output_shapes
:&: 4

_output_shapes
:&: 5

_output_shapes
::$6 

_output_shapes

:&:7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :,;(
&
_output_shapes
:&: <

_output_shapes
:&: =

_output_shapes
:&:,>(
&
_output_shapes
:&&: ?

_output_shapes
:&: @

_output_shapes
:&:,A(
&
_output_shapes
:&&: B

_output_shapes
:&: C

_output_shapes
:&: D

_output_shapes
::$E 

_output_shapes

:&:,F(
&
_output_shapes
:&: G

_output_shapes
:&: H

_output_shapes
:&:,I(
&
_output_shapes
:&&: J

_output_shapes
:&: K

_output_shapes
:&:,L(
&
_output_shapes
:&&: M

_output_shapes
:&: N

_output_shapes
:&: O

_output_shapes
::$P 

_output_shapes

:&:Q

_output_shapes
: 
ΧM

G__inference_sequential_4_layer_call_and_return_conditional_losses_58602

inputs
quantize_layer_58515
quantize_layer_58517
quant_conv2d_1_58520
quant_conv2d_1_58522
quant_conv2d_1_58524
quant_btch_norm_1_58527
quant_btch_norm_1_58529
quant_btch_norm_1_58531
quant_btch_norm_1_58533
quant_re_lu_6_58536
quant_re_lu_6_58538
quant_conv2d_2_58541
quant_conv2d_2_58543
quant_conv2d_2_58545
quant_btch_norm_2_58548
quant_btch_norm_2_58550
quant_btch_norm_2_58552
quant_btch_norm_2_58554
quant_re_lu_7_58557
quant_re_lu_7_58559
quant_conv2d_3_58562
quant_conv2d_3_58564
quant_conv2d_3_58566
quant_btch_norm_3_58569
quant_btch_norm_3_58571
quant_btch_norm_3_58573
quant_btch_norm_3_58575
quant_re_lu_8_58578
quant_re_lu_8_58580*
&quant_globalaveragepooling_layer_58583*
&quant_globalaveragepooling_layer_58585
quant_output_layer_58588
quant_output_layer_58590
quant_output_layer_58592
quant_output_layer_58594
quant_output_layer_58596
quant_output_layer_58598
identity’)quant_Btch_Norm-1/StatefulPartitionedCall’)quant_Btch_Norm-2/StatefulPartitionedCall’)quant_Btch_Norm-3/StatefulPartitionedCall’&quant_Conv2D-1/StatefulPartitionedCall’&quant_Conv2D-2/StatefulPartitionedCall’&quant_Conv2D-3/StatefulPartitionedCall’8quant_GlobalAveragePooling-Layer/StatefulPartitionedCall’*quant_Output-Layer/StatefulPartitionedCall’%quant_re_lu_6/StatefulPartitionedCall’%quant_re_lu_7/StatefulPartitionedCall’%quant_re_lu_8/StatefulPartitionedCall’&quantize_layer/StatefulPartitionedCall·
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_58515quantize_layer_58517*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quantize_layer_layer_call_and_return_conditional_losses_574272(
&quantize_layer/StatefulPartitionedCallψ
&quant_Conv2D-1/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_1_58520quant_conv2d_1_58522quant_conv2d_1_58524*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-1_layer_call_and_return_conditional_losses_574882(
&quant_Conv2D-1/StatefulPartitionedCall₯
)quant_Btch_Norm-1/StatefulPartitionedCallStatefulPartitionedCall/quant_Conv2D-1/StatefulPartitionedCall:output:0quant_btch_norm_1_58527quant_btch_norm_1_58529quant_btch_norm_1_58531quant_btch_norm_1_58533*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-1_layer_call_and_return_conditional_losses_575572+
)quant_Btch_Norm-1/StatefulPartitionedCallή
%quant_re_lu_6/StatefulPartitionedCallStatefulPartitionedCall2quant_Btch_Norm-1/StatefulPartitionedCall:output:0quant_re_lu_6_58536quant_re_lu_6_58538*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_6_layer_call_and_return_conditional_losses_576302'
%quant_re_lu_6/StatefulPartitionedCallχ
&quant_Conv2D-2/StatefulPartitionedCallStatefulPartitionedCall.quant_re_lu_6/StatefulPartitionedCall:output:0quant_conv2d_2_58541quant_conv2d_2_58543quant_conv2d_2_58545*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-2_layer_call_and_return_conditional_losses_576912(
&quant_Conv2D-2/StatefulPartitionedCall₯
)quant_Btch_Norm-2/StatefulPartitionedCallStatefulPartitionedCall/quant_Conv2D-2/StatefulPartitionedCall:output:0quant_btch_norm_2_58548quant_btch_norm_2_58550quant_btch_norm_2_58552quant_btch_norm_2_58554*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-2_layer_call_and_return_conditional_losses_577602+
)quant_Btch_Norm-2/StatefulPartitionedCallή
%quant_re_lu_7/StatefulPartitionedCallStatefulPartitionedCall2quant_Btch_Norm-2/StatefulPartitionedCall:output:0quant_re_lu_7_58557quant_re_lu_7_58559*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_7_layer_call_and_return_conditional_losses_578332'
%quant_re_lu_7/StatefulPartitionedCallχ
&quant_Conv2D-3/StatefulPartitionedCallStatefulPartitionedCall.quant_re_lu_7/StatefulPartitionedCall:output:0quant_conv2d_3_58562quant_conv2d_3_58564quant_conv2d_3_58566*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quant_Conv2D-3_layer_call_and_return_conditional_losses_578942(
&quant_Conv2D-3/StatefulPartitionedCall₯
)quant_Btch_Norm-3/StatefulPartitionedCallStatefulPartitionedCall/quant_Conv2D-3/StatefulPartitionedCall:output:0quant_btch_norm_3_58569quant_btch_norm_3_58571quant_btch_norm_3_58573quant_btch_norm_3_58575*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-3_layer_call_and_return_conditional_losses_579632+
)quant_Btch_Norm-3/StatefulPartitionedCallή
%quant_re_lu_8/StatefulPartitionedCallStatefulPartitionedCall2quant_Btch_Norm-3/StatefulPartitionedCall:output:0quant_re_lu_8_58578quant_re_lu_8_58580*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quant_re_lu_8_layer_call_and_return_conditional_losses_580362'
%quant_re_lu_8/StatefulPartitionedCall±
8quant_GlobalAveragePooling-Layer/StatefulPartitionedCallStatefulPartitionedCall.quant_re_lu_8/StatefulPartitionedCall:output:0&quant_globalaveragepooling_layer_58583&quant_globalaveragepooling_layer_58585*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *d
f_R]
[__inference_quant_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_580992:
8quant_GlobalAveragePooling-Layer/StatefulPartitionedCallξ
*quant_Output-Layer/StatefulPartitionedCallStatefulPartitionedCallAquant_GlobalAveragePooling-Layer/StatefulPartitionedCall:output:0quant_output_layer_58588quant_output_layer_58590quant_output_layer_58592quant_output_layer_58594quant_output_layer_58596quant_output_layer_58598*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_quant_Output-Layer_layer_call_and_return_conditional_losses_582002,
*quant_Output-Layer/StatefulPartitionedCall
IdentityIdentity3quant_Output-Layer/StatefulPartitionedCall:output:0*^quant_Btch_Norm-1/StatefulPartitionedCall*^quant_Btch_Norm-2/StatefulPartitionedCall*^quant_Btch_Norm-3/StatefulPartitionedCall'^quant_Conv2D-1/StatefulPartitionedCall'^quant_Conv2D-2/StatefulPartitionedCall'^quant_Conv2D-3/StatefulPartitionedCall9^quant_GlobalAveragePooling-Layer/StatefulPartitionedCall+^quant_Output-Layer/StatefulPartitionedCall&^quant_re_lu_6/StatefulPartitionedCall&^quant_re_lu_7/StatefulPartitionedCall&^quant_re_lu_8/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Δ
_input_shapes²
―:?????????1
:::::::::::::::::::::::::::::::::::::2V
)quant_Btch_Norm-1/StatefulPartitionedCall)quant_Btch_Norm-1/StatefulPartitionedCall2V
)quant_Btch_Norm-2/StatefulPartitionedCall)quant_Btch_Norm-2/StatefulPartitionedCall2V
)quant_Btch_Norm-3/StatefulPartitionedCall)quant_Btch_Norm-3/StatefulPartitionedCall2P
&quant_Conv2D-1/StatefulPartitionedCall&quant_Conv2D-1/StatefulPartitionedCall2P
&quant_Conv2D-2/StatefulPartitionedCall&quant_Conv2D-2/StatefulPartitionedCall2P
&quant_Conv2D-3/StatefulPartitionedCall&quant_Conv2D-3/StatefulPartitionedCall2t
8quant_GlobalAveragePooling-Layer/StatefulPartitionedCall8quant_GlobalAveragePooling-Layer/StatefulPartitionedCall2X
*quant_Output-Layer/StatefulPartitionedCall*quant_Output-Layer/StatefulPartitionedCall2N
%quant_re_lu_6/StatefulPartitionedCall%quant_re_lu_6/StatefulPartitionedCall2N
%quant_re_lu_7/StatefulPartitionedCall%quant_re_lu_7/StatefulPartitionedCall2N
%quant_re_lu_8/StatefulPartitionedCall%quant_re_lu_8/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs


.__inference_quantize_layer_layer_call_fn_59360

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????1
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_quantize_layer_layer_call_and_return_conditional_losses_574272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????1
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????1
::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs
«)
Ι
H__inference_quant_re_lu_7_layer_call_and_return_conditional_losses_59684

inputs:
6movingavgquantize_assignminema_readvariableop_resource:
6movingavgquantize_assignmaxema_readvariableop_resource
identity’2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp’2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpV
ReluReluinputs*
T0*/
_output_shapes
:?????????&2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayΝ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subΗ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul²
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayΝ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subΗ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul²
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ό
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsω
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:03^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&::2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
π
¨
,__inference_sequential_4_layer_call_fn_59233

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35
identity’StatefulPartitionedCallΎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
 #*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_584332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Δ
_input_shapes²
―:?????????1
:::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs

?
F__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_57161

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&:::::i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs

ρ
M__inference_quant_Output-Layer_layer_call_and_return_conditional_losses_58200

inputsB
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resourceD
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resourceD
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource#
biasadd_readvariableop_resourceE
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resourceG
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource
identityν
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:&*
dtype027
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpλ
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1λ
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ΰ
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:&*
narrow_range(2(
&LastValueQuant/FakeQuantWithMinMaxVars
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddξ
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpτ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1²
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&:::::::O K
'
_output_shapes
:?????????&
 
_user_specified_nameinputs
»
χ
H__inference_quant_re_lu_7_layer_call_and_return_conditional_losses_57833

inputsE
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resourceG
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????&2
Reluξ
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpτ
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ό
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&:::W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
«)
Ι
H__inference_quant_re_lu_7_layer_call_and_return_conditional_losses_57823

inputs:
6movingavgquantize_assignminema_readvariableop_resource:
6movingavgquantize_assignmaxema_readvariableop_resource
identity’2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp’2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpV
ReluReluinputs*
T0*/
_output_shapes
:?????????&2
Relu
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y­
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y­
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMinEma/decayΝ
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/subΗ
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul²
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o:2&
$MovingAvgQuantize/AssignMaxEma/decayΝ
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpΖ
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/subΗ
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul²
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Ό
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&2+
)MovingAvgQuantize/FakeQuantWithMinMaxVarsω
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:03^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????&::2h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
ό
©
L__inference_quant_Btch_Norm-2_layer_call_and_return_conditional_losses_59614

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ψ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
Λ
€
1__inference_quant_Btch_Norm-1_layer_call_fn_59469

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_quant_Btch_Norm-1_layer_call_and_return_conditional_losses_575392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs


+__inference_Btch_Norm-1_layer_call_fn_60105

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity’StatefulPartitionedCall¨
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
GPU 2J 8 *O
fJRH
F__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_571302
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
½Ν
-
!__inference__traced_restore_60759
file_prefix6
2assignvariableop_quantize_layer_quantize_layer_min8
4assignvariableop_1_quantize_layer_quantize_layer_max4
0assignvariableop_2_quantize_layer_optimizer_step4
0assignvariableop_3_quant_conv2d_1_optimizer_step0
,assignvariableop_4_quant_conv2d_1_kernel_min0
,assignvariableop_5_quant_conv2d_1_kernel_max7
3assignvariableop_6_quant_btch_norm_1_optimizer_step3
/assignvariableop_7_quant_re_lu_6_optimizer_step/
+assignvariableop_8_quant_re_lu_6_output_min/
+assignvariableop_9_quant_re_lu_6_output_max5
1assignvariableop_10_quant_conv2d_2_optimizer_step1
-assignvariableop_11_quant_conv2d_2_kernel_min1
-assignvariableop_12_quant_conv2d_2_kernel_max8
4assignvariableop_13_quant_btch_norm_2_optimizer_step4
0assignvariableop_14_quant_re_lu_7_optimizer_step0
,assignvariableop_15_quant_re_lu_7_output_min0
,assignvariableop_16_quant_re_lu_7_output_max5
1assignvariableop_17_quant_conv2d_3_optimizer_step1
-assignvariableop_18_quant_conv2d_3_kernel_min1
-assignvariableop_19_quant_conv2d_3_kernel_max8
4assignvariableop_20_quant_btch_norm_3_optimizer_step4
0assignvariableop_21_quant_re_lu_8_optimizer_step0
,assignvariableop_22_quant_re_lu_8_output_min0
,assignvariableop_23_quant_re_lu_8_output_maxG
Cassignvariableop_24_quant_globalaveragepooling_layer_optimizer_stepC
?assignvariableop_25_quant_globalaveragepooling_layer_output_minC
?assignvariableop_26_quant_globalaveragepooling_layer_output_max9
5assignvariableop_27_quant_output_layer_optimizer_step5
1assignvariableop_28_quant_output_layer_kernel_min5
1assignvariableop_29_quant_output_layer_kernel_max>
:assignvariableop_30_quant_output_layer_post_activation_min>
:assignvariableop_31_quant_output_layer_post_activation_max!
assignvariableop_32_adam_iter#
assignvariableop_33_adam_beta_1#
assignvariableop_34_adam_beta_2"
assignvariableop_35_adam_decay*
&assignvariableop_36_adam_learning_rate'
#assignvariableop_37_conv2d_1_kernel)
%assignvariableop_38_btch_norm_1_gamma(
$assignvariableop_39_btch_norm_1_beta/
+assignvariableop_40_btch_norm_1_moving_mean3
/assignvariableop_41_btch_norm_1_moving_variance'
#assignvariableop_42_conv2d_2_kernel)
%assignvariableop_43_btch_norm_2_gamma(
$assignvariableop_44_btch_norm_2_beta/
+assignvariableop_45_btch_norm_2_moving_mean3
/assignvariableop_46_btch_norm_2_moving_variance'
#assignvariableop_47_conv2d_3_kernel)
%assignvariableop_48_btch_norm_3_gamma(
$assignvariableop_49_btch_norm_3_beta/
+assignvariableop_50_btch_norm_3_moving_mean3
/assignvariableop_51_btch_norm_3_moving_variance)
%assignvariableop_52_output_layer_bias+
'assignvariableop_53_output_layer_kernel
assignvariableop_54_total
assignvariableop_55_count
assignvariableop_56_total_1
assignvariableop_57_count_1.
*assignvariableop_58_adam_conv2d_1_kernel_m0
,assignvariableop_59_adam_btch_norm_1_gamma_m/
+assignvariableop_60_adam_btch_norm_1_beta_m.
*assignvariableop_61_adam_conv2d_2_kernel_m0
,assignvariableop_62_adam_btch_norm_2_gamma_m/
+assignvariableop_63_adam_btch_norm_2_beta_m.
*assignvariableop_64_adam_conv2d_3_kernel_m0
,assignvariableop_65_adam_btch_norm_3_gamma_m/
+assignvariableop_66_adam_btch_norm_3_beta_m0
,assignvariableop_67_adam_output_layer_bias_m2
.assignvariableop_68_adam_output_layer_kernel_m.
*assignvariableop_69_adam_conv2d_1_kernel_v0
,assignvariableop_70_adam_btch_norm_1_gamma_v/
+assignvariableop_71_adam_btch_norm_1_beta_v.
*assignvariableop_72_adam_conv2d_2_kernel_v0
,assignvariableop_73_adam_btch_norm_2_gamma_v/
+assignvariableop_74_adam_btch_norm_2_beta_v.
*assignvariableop_75_adam_conv2d_3_kernel_v0
,assignvariableop_76_adam_btch_norm_3_gamma_v/
+assignvariableop_77_adam_btch_norm_3_beta_v0
,assignvariableop_78_adam_output_layer_bias_v2
.assignvariableop_79_adam_output_layer_kernel_v
identity_81’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_45’AssignVariableOp_46’AssignVariableOp_47’AssignVariableOp_48’AssignVariableOp_49’AssignVariableOp_5’AssignVariableOp_50’AssignVariableOp_51’AssignVariableOp_52’AssignVariableOp_53’AssignVariableOp_54’AssignVariableOp_55’AssignVariableOp_56’AssignVariableOp_57’AssignVariableOp_58’AssignVariableOp_59’AssignVariableOp_6’AssignVariableOp_60’AssignVariableOp_61’AssignVariableOp_62’AssignVariableOp_63’AssignVariableOp_64’AssignVariableOp_65’AssignVariableOp_66’AssignVariableOp_67’AssignVariableOp_68’AssignVariableOp_69’AssignVariableOp_7’AssignVariableOp_70’AssignVariableOp_71’AssignVariableOp_72’AssignVariableOp_73’AssignVariableOp_74’AssignVariableOp_75’AssignVariableOp_76’AssignVariableOp_77’AssignVariableOp_78’AssignVariableOp_79’AssignVariableOp_8’AssignVariableOp_9Ϋ%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*η$
valueέ$BΪ$QBBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/output_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/kernel_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-6/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/output_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-7/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/kernel_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-8/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-9/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/output_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/output_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-10/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-10/output_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-10/output_max/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-11/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-11/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-11/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-11/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-11/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/42/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/43/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names³
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:Q*
dtype0*·
value­BͺQB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesΓ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ϊ
_output_shapesΗ
Δ:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*_
dtypesU
S2Q	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity±
AssignVariableOpAssignVariableOp2assignvariableop_quantize_layer_quantize_layer_minIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ή
AssignVariableOp_1AssignVariableOp4assignvariableop_1_quantize_layer_quantize_layer_maxIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2΅
AssignVariableOp_2AssignVariableOp0assignvariableop_2_quantize_layer_optimizer_stepIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3΅
AssignVariableOp_3AssignVariableOp0assignvariableop_3_quant_conv2d_1_optimizer_stepIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4±
AssignVariableOp_4AssignVariableOp,assignvariableop_4_quant_conv2d_1_kernel_minIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5±
AssignVariableOp_5AssignVariableOp,assignvariableop_5_quant_conv2d_1_kernel_maxIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Έ
AssignVariableOp_6AssignVariableOp3assignvariableop_6_quant_btch_norm_1_optimizer_stepIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7΄
AssignVariableOp_7AssignVariableOp/assignvariableop_7_quant_re_lu_6_optimizer_stepIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8°
AssignVariableOp_8AssignVariableOp+assignvariableop_8_quant_re_lu_6_output_minIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9°
AssignVariableOp_9AssignVariableOp+assignvariableop_9_quant_re_lu_6_output_maxIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ή
AssignVariableOp_10AssignVariableOp1assignvariableop_10_quant_conv2d_2_optimizer_stepIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11΅
AssignVariableOp_11AssignVariableOp-assignvariableop_11_quant_conv2d_2_kernel_minIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12΅
AssignVariableOp_12AssignVariableOp-assignvariableop_12_quant_conv2d_2_kernel_maxIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ό
AssignVariableOp_13AssignVariableOp4assignvariableop_13_quant_btch_norm_2_optimizer_stepIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Έ
AssignVariableOp_14AssignVariableOp0assignvariableop_14_quant_re_lu_7_optimizer_stepIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15΄
AssignVariableOp_15AssignVariableOp,assignvariableop_15_quant_re_lu_7_output_minIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16΄
AssignVariableOp_16AssignVariableOp,assignvariableop_16_quant_re_lu_7_output_maxIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ή
AssignVariableOp_17AssignVariableOp1assignvariableop_17_quant_conv2d_3_optimizer_stepIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18΅
AssignVariableOp_18AssignVariableOp-assignvariableop_18_quant_conv2d_3_kernel_minIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19΅
AssignVariableOp_19AssignVariableOp-assignvariableop_19_quant_conv2d_3_kernel_maxIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ό
AssignVariableOp_20AssignVariableOp4assignvariableop_20_quant_btch_norm_3_optimizer_stepIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Έ
AssignVariableOp_21AssignVariableOp0assignvariableop_21_quant_re_lu_8_optimizer_stepIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22΄
AssignVariableOp_22AssignVariableOp,assignvariableop_22_quant_re_lu_8_output_minIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23΄
AssignVariableOp_23AssignVariableOp,assignvariableop_23_quant_re_lu_8_output_maxIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Λ
AssignVariableOp_24AssignVariableOpCassignvariableop_24_quant_globalaveragepooling_layer_optimizer_stepIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Η
AssignVariableOp_25AssignVariableOp?assignvariableop_25_quant_globalaveragepooling_layer_output_minIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Η
AssignVariableOp_26AssignVariableOp?assignvariableop_26_quant_globalaveragepooling_layer_output_maxIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27½
AssignVariableOp_27AssignVariableOp5assignvariableop_27_quant_output_layer_optimizer_stepIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Ή
AssignVariableOp_28AssignVariableOp1assignvariableop_28_quant_output_layer_kernel_minIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ή
AssignVariableOp_29AssignVariableOp1assignvariableop_29_quant_output_layer_kernel_maxIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Β
AssignVariableOp_30AssignVariableOp:assignvariableop_30_quant_output_layer_post_activation_minIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Β
AssignVariableOp_31AssignVariableOp:assignvariableop_31_quant_output_layer_post_activation_maxIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_32₯
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_iterIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33§
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_beta_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34§
AssignVariableOp_34AssignVariableOpassignvariableop_34_adam_beta_2Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¦
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_decayIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_learning_rateIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37«
AssignVariableOp_37AssignVariableOp#assignvariableop_37_conv2d_1_kernelIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38­
AssignVariableOp_38AssignVariableOp%assignvariableop_38_btch_norm_1_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¬
AssignVariableOp_39AssignVariableOp$assignvariableop_39_btch_norm_1_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40³
AssignVariableOp_40AssignVariableOp+assignvariableop_40_btch_norm_1_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41·
AssignVariableOp_41AssignVariableOp/assignvariableop_41_btch_norm_1_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42«
AssignVariableOp_42AssignVariableOp#assignvariableop_42_conv2d_2_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43­
AssignVariableOp_43AssignVariableOp%assignvariableop_43_btch_norm_2_gammaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¬
AssignVariableOp_44AssignVariableOp$assignvariableop_44_btch_norm_2_betaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45³
AssignVariableOp_45AssignVariableOp+assignvariableop_45_btch_norm_2_moving_meanIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46·
AssignVariableOp_46AssignVariableOp/assignvariableop_46_btch_norm_2_moving_varianceIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47«
AssignVariableOp_47AssignVariableOp#assignvariableop_47_conv2d_3_kernelIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48­
AssignVariableOp_48AssignVariableOp%assignvariableop_48_btch_norm_3_gammaIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49¬
AssignVariableOp_49AssignVariableOp$assignvariableop_49_btch_norm_3_betaIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50³
AssignVariableOp_50AssignVariableOp+assignvariableop_50_btch_norm_3_moving_meanIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51·
AssignVariableOp_51AssignVariableOp/assignvariableop_51_btch_norm_3_moving_varianceIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52­
AssignVariableOp_52AssignVariableOp%assignvariableop_52_output_layer_biasIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53―
AssignVariableOp_53AssignVariableOp'assignvariableop_53_output_layer_kernelIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54‘
AssignVariableOp_54AssignVariableOpassignvariableop_54_totalIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55‘
AssignVariableOp_55AssignVariableOpassignvariableop_55_countIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56£
AssignVariableOp_56AssignVariableOpassignvariableop_56_total_1Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57£
AssignVariableOp_57AssignVariableOpassignvariableop_57_count_1Identity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58²
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_conv2d_1_kernel_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59΄
AssignVariableOp_59AssignVariableOp,assignvariableop_59_adam_btch_norm_1_gamma_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60³
AssignVariableOp_60AssignVariableOp+assignvariableop_60_adam_btch_norm_1_beta_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61²
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_conv2d_2_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62΄
AssignVariableOp_62AssignVariableOp,assignvariableop_62_adam_btch_norm_2_gamma_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63³
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_btch_norm_2_beta_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64²
AssignVariableOp_64AssignVariableOp*assignvariableop_64_adam_conv2d_3_kernel_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65΄
AssignVariableOp_65AssignVariableOp,assignvariableop_65_adam_btch_norm_3_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66³
AssignVariableOp_66AssignVariableOp+assignvariableop_66_adam_btch_norm_3_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67΄
AssignVariableOp_67AssignVariableOp,assignvariableop_67_adam_output_layer_bias_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68Ά
AssignVariableOp_68AssignVariableOp.assignvariableop_68_adam_output_layer_kernel_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69²
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_conv2d_1_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70΄
AssignVariableOp_70AssignVariableOp,assignvariableop_70_adam_btch_norm_1_gamma_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71³
AssignVariableOp_71AssignVariableOp+assignvariableop_71_adam_btch_norm_1_beta_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72²
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_conv2d_2_kernel_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73΄
AssignVariableOp_73AssignVariableOp,assignvariableop_73_adam_btch_norm_2_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74³
AssignVariableOp_74AssignVariableOp+assignvariableop_74_adam_btch_norm_2_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75²
AssignVariableOp_75AssignVariableOp*assignvariableop_75_adam_conv2d_3_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76΄
AssignVariableOp_76AssignVariableOp,assignvariableop_76_adam_btch_norm_3_gamma_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77³
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_btch_norm_3_beta_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78΄
AssignVariableOp_78AssignVariableOp,assignvariableop_78_adam_output_layer_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79Ά
AssignVariableOp_79AssignVariableOp.assignvariableop_79_adam_output_layer_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_799
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpΎ
Identity_80Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_80±
Identity_81IdentityIdentity_80:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_81"#
identity_81Identity_81:output:0*Χ
_input_shapesΕ
Β: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

?
F__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_60156

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????&:&:&:&:&:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????&2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????&:::::i e
A
_output_shapes/
-:+???????????????????????????&
 
_user_specified_nameinputs
ζ#
±
I__inference_quant_Conv2D-2_layer_call_and_return_conditional_losses_59560

inputs3
/lastvaluequant_batchmin_readvariableop_resource)
%lastvaluequant_assignminlast_resource)
%lastvaluequant_assignmaxlast_resource
identity’LastValueQuant/AssignMaxLast’LastValueQuant/AssignMinLastΘ
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOp«
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indicesΒ
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:&2
LastValueQuant/BatchMinΘ
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOp«
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indicesΒ
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:&2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2
LastValueQuant/truediv/y₯
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:&2
LastValueQuant/truediv
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:&2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Ώ2
LastValueQuant/mul/y
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:&2
LastValueQuant/mul
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:&2
LastValueQuant/MaximumΆ
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLastΆ
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLastϊ
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:&&*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:&*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2€
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&&*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannelΑ
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
Conv2D©
IdentityIdentityConv2D:output:0^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????&:::2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs
¨
Ψ
G__inference_sequential_4_layer_call_and_return_conditional_losses_59154

inputsT
Pquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resourceV
Rquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource[
Wquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource]
Yquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource]
Yquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource-
)quant_btch_norm_1_readvariableop_resource/
+quant_btch_norm_1_readvariableop_1_resource>
:quant_btch_norm_1_fusedbatchnormv3_readvariableop_resource@
<quant_btch_norm_1_fusedbatchnormv3_readvariableop_1_resourceS
Oquant_re_lu_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resourceU
Qquant_re_lu_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource[
Wquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource]
Yquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource]
Yquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource-
)quant_btch_norm_2_readvariableop_resource/
+quant_btch_norm_2_readvariableop_1_resource>
:quant_btch_norm_2_fusedbatchnormv3_readvariableop_resource@
<quant_btch_norm_2_fusedbatchnormv3_readvariableop_1_resourceS
Oquant_re_lu_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_resourceU
Qquant_re_lu_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource[
Wquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource]
Yquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource]
Yquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource-
)quant_btch_norm_3_readvariableop_resource/
+quant_btch_norm_3_readvariableop_1_resource>
:quant_btch_norm_3_fusedbatchnormv3_readvariableop_resource@
<quant_btch_norm_3_fusedbatchnormv3_readvariableop_1_resourceS
Oquant_re_lu_8_movingavgquantize_fakequantwithminmaxvars_readvariableop_resourceU
Qquant_re_lu_8_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resourcef
bquant_globalaveragepooling_layer_movingavgquantize_fakequantwithminmaxvars_readvariableop_resourceh
dquant_globalaveragepooling_layer_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resourceU
Qquant_output_layer_lastvaluequant_fakequantwithminmaxvars_readvariableop_resourceW
Squant_output_layer_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resourceW
Squant_output_layer_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource6
2quant_output_layer_biasadd_readvariableop_resourceX
Tquant_output_layer_movingavgquantize_fakequantwithminmaxvars_readvariableop_resourceZ
Vquant_output_layer_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource
identity
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02I
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp‘
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02K
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1μ
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsOquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????1
2:
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsΐ
Nquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:&*
dtype02P
Nquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpΊ
Pquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:&*
dtype02R
Pquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Ί
Pquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_1_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:&*
dtype02R
Pquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ο
?quant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&*
narrow_range(2A
?quant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannelͺ
quant_Conv2D-1/Conv2DConv2DBquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_Conv2D-1/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
quant_Conv2D-1/Conv2Dͺ
 quant_Btch_Norm-1/ReadVariableOpReadVariableOp)quant_btch_norm_1_readvariableop_resource*
_output_shapes
:&*
dtype02"
 quant_Btch_Norm-1/ReadVariableOp°
"quant_Btch_Norm-1/ReadVariableOp_1ReadVariableOp+quant_btch_norm_1_readvariableop_1_resource*
_output_shapes
:&*
dtype02$
"quant_Btch_Norm-1/ReadVariableOp_1έ
1quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOpReadVariableOp:quant_btch_norm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype023
1quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOpγ
3quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<quant_btch_norm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype025
3quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1Ξ
"quant_Btch_Norm-1/FusedBatchNormV3FusedBatchNormV3quant_Conv2D-1/Conv2D:output:0(quant_Btch_Norm-1/ReadVariableOp:value:0*quant_Btch_Norm-1/ReadVariableOp_1:value:09quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp:value:0;quant_Btch_Norm-1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
is_training( 2$
"quant_Btch_Norm-1/FusedBatchNormV3
quant_re_lu_6/ReluRelu&quant_Btch_Norm-1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&2
quant_re_lu_6/Relu
Fquant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpOquant_re_lu_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02H
Fquant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
Hquant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpQquant_re_lu_6_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02J
Hquant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
7quant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars quant_re_lu_6/Relu:activations:0Nquant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&29
7quant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVarsΐ
Nquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:&&*
dtype02P
Nquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpΊ
Pquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:&*
dtype02R
Pquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Ί
Pquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_2_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:&*
dtype02R
Pquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ο
?quant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&&*
narrow_range(2A
?quant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel©
quant_Conv2D-2/Conv2DConv2DAquant_re_lu_6/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_Conv2D-2/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
quant_Conv2D-2/Conv2Dͺ
 quant_Btch_Norm-2/ReadVariableOpReadVariableOp)quant_btch_norm_2_readvariableop_resource*
_output_shapes
:&*
dtype02"
 quant_Btch_Norm-2/ReadVariableOp°
"quant_Btch_Norm-2/ReadVariableOp_1ReadVariableOp+quant_btch_norm_2_readvariableop_1_resource*
_output_shapes
:&*
dtype02$
"quant_Btch_Norm-2/ReadVariableOp_1έ
1quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOpReadVariableOp:quant_btch_norm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype023
1quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOpγ
3quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<quant_btch_norm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype025
3quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1Ξ
"quant_Btch_Norm-2/FusedBatchNormV3FusedBatchNormV3quant_Conv2D-2/Conv2D:output:0(quant_Btch_Norm-2/ReadVariableOp:value:0*quant_Btch_Norm-2/ReadVariableOp_1:value:09quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp:value:0;quant_Btch_Norm-2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
is_training( 2$
"quant_Btch_Norm-2/FusedBatchNormV3
quant_re_lu_7/ReluRelu&quant_Btch_Norm-2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&2
quant_re_lu_7/Relu
Fquant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpOquant_re_lu_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02H
Fquant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
Hquant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpQquant_re_lu_7_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02J
Hquant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
7quant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars quant_re_lu_7/Relu:activations:0Nquant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&29
7quant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVarsΐ
Nquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpWquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:&&*
dtype02P
Nquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpΊ
Pquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpYquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:&*
dtype02R
Pquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Ί
Pquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpYquant_conv2d_3_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:&*
dtype02R
Pquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ο
?quant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelVquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Xquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Xquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:&&*
narrow_range(2A
?quant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel©
quant_Conv2D-3/Conv2DConv2DAquant_re_lu_7/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Iquant_Conv2D-3/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????&*
paddingVALID*
strides
2
quant_Conv2D-3/Conv2Dͺ
 quant_Btch_Norm-3/ReadVariableOpReadVariableOp)quant_btch_norm_3_readvariableop_resource*
_output_shapes
:&*
dtype02"
 quant_Btch_Norm-3/ReadVariableOp°
"quant_Btch_Norm-3/ReadVariableOp_1ReadVariableOp+quant_btch_norm_3_readvariableop_1_resource*
_output_shapes
:&*
dtype02$
"quant_Btch_Norm-3/ReadVariableOp_1έ
1quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOpReadVariableOp:quant_btch_norm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype023
1quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOpγ
3quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<quant_btch_norm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype025
3quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1Ξ
"quant_Btch_Norm-3/FusedBatchNormV3FusedBatchNormV3quant_Conv2D-3/Conv2D:output:0(quant_Btch_Norm-3/ReadVariableOp:value:0*quant_Btch_Norm-3/ReadVariableOp_1:value:09quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp:value:0;quant_Btch_Norm-3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
is_training( 2$
"quant_Btch_Norm-3/FusedBatchNormV3
quant_re_lu_8/ReluRelu&quant_Btch_Norm-3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????&2
quant_re_lu_8/Relu
Fquant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpOquant_re_lu_8_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02H
Fquant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
Hquant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpQquant_re_lu_8_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02J
Hquant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
7quant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars quant_re_lu_8/Relu:activations:0Nquant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Pquant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????&29
7quant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVarsΓ
7quant_GlobalAveragePooling-Layer/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      29
7quant_GlobalAveragePooling-Layer/Mean/reduction_indices
%quant_GlobalAveragePooling-Layer/MeanMeanAquant_re_lu_8/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0@quant_GlobalAveragePooling-Layer/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:?????????&2'
%quant_GlobalAveragePooling-Layer/MeanΡ
Yquant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpbquant_globalaveragepooling_layer_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02[
Yquant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpΧ
[quant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpdquant_globalaveragepooling_layer_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02]
[quant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Τ
Jquant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars.quant_GlobalAveragePooling-Layer/Mean:output:0aquant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0cquant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????&2L
Jquant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars¦
Hquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpQquant_output_layer_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes

:&*
dtype02J
Hquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp€
Jquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpSquant_output_layer_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02L
Jquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1€
Jquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpSquant_output_layer_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype02L
Jquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Ώ
9quant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsPquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Rquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Rquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes

:&*
narrow_range(2;
9quant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars
quant_Output-Layer/MatMulMatMulTquant_GlobalAveragePooling-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0Cquant_Output-Layer/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2
quant_Output-Layer/MatMulΕ
)quant_Output-Layer/BiasAdd/ReadVariableOpReadVariableOp2quant_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)quant_Output-Layer/BiasAdd/ReadVariableOpΝ
quant_Output-Layer/BiasAddBiasAdd#quant_Output-Layer/MatMul:product:01quant_Output-Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
quant_Output-Layer/BiasAdd§
Kquant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpTquant_output_layer_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02M
Kquant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp­
Mquant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpVquant_output_layer_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02O
Mquant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1
<quant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars#quant_Output-Layer/BiasAdd:output:0Squant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Uquant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????2>
<quant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars
IdentityIdentityFquant_Output-Layer/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Δ
_input_shapes²
―:?????????1
::::::::::::::::::::::::::::::::::::::W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs
Έ 
’
I__inference_quantize_layer_layer_call_and_return_conditional_losses_59333

inputs5
1allvaluesquantize_minimum_readvariableop_resource5
1allvaluesquantize_maximum_readvariableop_resource
identity’#AllValuesQuantize/AssignMaxAllValue’#AllValuesQuantize/AssignMinAllValue
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
AllValuesQuantize/Const
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/BatchMin
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
AllValuesQuantize/Const_1
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/BatchMaxΎ
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype02*
(AllValuesQuantize/Minimum/ReadVariableOpΉ
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Minimum
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
AllValuesQuantize/Minimum_1/y­
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Minimum_1Ύ
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype02*
(AllValuesQuantize/Maximum/ReadVariableOpΉ
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Maximum
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
AllValuesQuantize/Maximum_1/y­
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Maximum_1
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype02%
#AllValuesQuantize/AssignMinAllValue
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype02%
#AllValuesQuantize/AssignMaxAllValue
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype02:
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype02<
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1°
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????1
2+
)AllValuesQuantize/FakeQuantWithMinMaxVarsΫ
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue*
T0*/
_output_shapes
:?????????1
2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????1
::2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue:W S
/
_output_shapes
:?????????1

 
_user_specified_nameinputs
’
°
,__inference_sequential_4_layer_call_fn_58679
conv2d_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35
identity’StatefulPartitionedCallΰ
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*G
_read_only_resource_inputs)
'%	
 !"#$%*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_586022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*Δ
_input_shapes²
―:?????????1
:::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:?????????1

(
_user_specified_nameConv2D-1_input
ό
©
L__inference_quant_Btch_Norm-1_layer_call_and_return_conditional_losses_57539

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity’AssignNewValue’AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:&*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:&*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:&*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:&*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ψ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????&:&:&:&:&:*
epsilon%o:*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:?????????&2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:?????????&::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:?????????&
 
_user_specified_nameinputs"ΈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Λ
serving_default·
Q
Conv2D-1_input?
 serving_default_Conv2D-1_input:0?????????1
F
quant_Output-Layer0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:Ηή

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
	layer_with_weights-8
	layer-8

layer_with_weights-9

layer-9
layer_with_weights-10
layer-10
layer_with_weights-11
layer-11
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+ϊ&call_and_return_all_conditional_losses
ϋ__call__
ό_default_save_signature"ά
_tf_keras_sequentialΌ{"class_name": "Sequential", "name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 49, 10, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Conv2D-1_input"}}, {"class_name": "QuantizeLayer", "config": {"name": "quantize_layer", "trainable": true, "dtype": "float32", "quantizer": {"class_name": "AllValuesQuantizer", "config": {"num_bits": 8, "per_axis": false, "symmetric": false, "narrow_range": false}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_Conv2D-1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "Conv2D-1", "trainable": true, "dtype": "float32", "filters": 38, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": {"class_name": "NoOpActivation", "config": {}}}}, "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_Btch_Norm-1", "trainable": true, "dtype": "float32", "layer": {"class_name": "BatchNormalization", "config": {"name": "Btch_Norm-1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_re_lu_6", "trainable": true, "dtype": "float32", "layer": {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": true}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_Conv2D-2", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "Conv2D-2", "trainable": true, "dtype": "float32", "filters": 38, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": {"class_name": "NoOpActivation", "config": {}}}}, "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_Btch_Norm-2", "trainable": true, "dtype": "float32", "layer": {"class_name": "BatchNormalization", "config": {"name": "Btch_Norm-2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_re_lu_7", "trainable": true, "dtype": "float32", "layer": {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": true}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_Conv2D-3", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "Conv2D-3", "trainable": true, "dtype": "float32", "filters": 38, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": {"class_name": "NoOpActivation", "config": {}}}}, "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_Btch_Norm-3", "trainable": true, "dtype": "float32", "layer": {"class_name": "BatchNormalization", "config": {"name": "Btch_Norm-3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_re_lu_8", "trainable": true, "dtype": "float32", "layer": {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": true}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_GlobalAveragePooling-Layer", "trainable": true, "dtype": "float32", "layer": {"class_name": "GlobalAveragePooling2D", "config": {"name": "GlobalAveragePooling-Layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": true}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_Output-Layer", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "Output-Layer", "trainable": true, "dtype": "float32", "units": 8, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "linear"}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49, 10, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 49, 10, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Conv2D-1_input"}}, {"class_name": "QuantizeLayer", "config": {"name": "quantize_layer", "trainable": true, "dtype": "float32", "quantizer": {"class_name": "AllValuesQuantizer", "config": {"num_bits": 8, "per_axis": false, "symmetric": false, "narrow_range": false}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_Conv2D-1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "Conv2D-1", "trainable": true, "dtype": "float32", "filters": 38, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": {"class_name": "NoOpActivation", "config": {}}}}, "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_Btch_Norm-1", "trainable": true, "dtype": "float32", "layer": {"class_name": "BatchNormalization", "config": {"name": "Btch_Norm-1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_re_lu_6", "trainable": true, "dtype": "float32", "layer": {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": true}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_Conv2D-2", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "Conv2D-2", "trainable": true, "dtype": "float32", "filters": 38, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": {"class_name": "NoOpActivation", "config": {}}}}, "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_Btch_Norm-2", "trainable": true, "dtype": "float32", "layer": {"class_name": "BatchNormalization", "config": {"name": "Btch_Norm-2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_re_lu_7", "trainable": true, "dtype": "float32", "layer": {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": true}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_Conv2D-3", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "Conv2D-3", "trainable": true, "dtype": "float32", "filters": 38, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": {"class_name": "NoOpActivation", "config": {}}}}, "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_Btch_Norm-3", "trainable": true, "dtype": "float32", "layer": {"class_name": "BatchNormalization", "config": {"name": "Btch_Norm-3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_re_lu_8", "trainable": true, "dtype": "float32", "layer": {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": true}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_GlobalAveragePooling-Layer", "trainable": true, "dtype": "float32", "layer": {"class_name": "GlobalAveragePooling2D", "config": {"name": "GlobalAveragePooling-Layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": true}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_Output-Layer", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "Output-Layer", "trainable": true, "dtype": "float32", "units": 8, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "linear"}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}}]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": [{"class_name": "SparseCategoricalAccuracy", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}}], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ϋ
quantize_layer_min
quantize_layer_max
quantizer_vars
optimizer_step
	variables
regularization_losses
trainable_variables
	keras_api
+ύ&call_and_return_all_conditional_losses
ώ__call__"
_tf_keras_layerψ{"class_name": "QuantizeLayer", "name": "quantize_layer", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "quantize_layer", "trainable": true, "dtype": "float32", "quantizer": {"class_name": "AllValuesQuantizer", "config": {"num_bits": 8, "per_axis": false, "symmetric": false, "narrow_range": false}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49, 10, 1]}}
ΰ
	layer
optimizer_step
_weight_vars

kernel_min

kernel_max
 _quantize_activations
!_output_quantizers
"	variables
#regularization_losses
$trainable_variables
%	keras_api
+?&call_and_return_all_conditional_losses
__call__"Λ

_tf_keras_layer±
{"class_name": "QuantizeWrapper", "name": "quant_Conv2D-1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "quant_Conv2D-1", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "Conv2D-1", "trainable": true, "dtype": "float32", "filters": 38, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": {"class_name": "NoOpActivation", "config": {}}}}, "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49, 10, 1]}}
χ	
	&layer
'optimizer_step
(_weight_vars
)_quantize_activations
*_output_quantizers
+	variables
,regularization_losses
-trainable_variables
.	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerθ{"class_name": "QuantizeWrapper", "name": "quant_Btch_Norm-1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "quant_Btch_Norm-1", "trainable": true, "dtype": "float32", "layer": {"class_name": "BatchNormalization", "config": {"name": "Btch_Norm-1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 8, 38]}}
λ
	/layer
0optimizer_step
1_weight_vars
2_quantize_activations
3_output_quantizers
4
output_min
5
output_max
6_output_quantizer_vars
7	variables
8regularization_losses
9trainable_variables
:	keras_api
+&call_and_return_all_conditional_losses
__call__"Ί
_tf_keras_layer {"class_name": "QuantizeWrapper", "name": "quant_re_lu_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "quant_re_lu_6", "trainable": true, "dtype": "float32", "layer": {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": true}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 8, 38]}}
ΰ
	;layer
<optimizer_step
=_weight_vars
>
kernel_min
?
kernel_max
@_quantize_activations
A_output_quantizers
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
+&call_and_return_all_conditional_losses
__call__"Λ

_tf_keras_layer±
{"class_name": "QuantizeWrapper", "name": "quant_Conv2D-2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "quant_Conv2D-2", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "Conv2D-2", "trainable": true, "dtype": "float32", "filters": 38, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": {"class_name": "NoOpActivation", "config": {}}}}, "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 8, 38]}}
χ	
	Flayer
Goptimizer_step
H_weight_vars
I_quantize_activations
J_output_quantizers
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerθ{"class_name": "QuantizeWrapper", "name": "quant_Btch_Norm-2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "quant_Btch_Norm-2", "trainable": true, "dtype": "float32", "layer": {"class_name": "BatchNormalization", "config": {"name": "Btch_Norm-2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 6, 38]}}
λ
	Olayer
Poptimizer_step
Q_weight_vars
R_quantize_activations
S_output_quantizers
T
output_min
U
output_max
V_output_quantizer_vars
W	variables
Xregularization_losses
Ytrainable_variables
Z	keras_api
+&call_and_return_all_conditional_losses
__call__"Ί
_tf_keras_layer {"class_name": "QuantizeWrapper", "name": "quant_re_lu_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "quant_re_lu_7", "trainable": true, "dtype": "float32", "layer": {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": true}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 6, 38]}}
ΰ
	[layer
\optimizer_step
]_weight_vars
^
kernel_min
_
kernel_max
`_quantize_activations
a_output_quantizers
b	variables
cregularization_losses
dtrainable_variables
e	keras_api
+&call_and_return_all_conditional_losses
__call__"Λ

_tf_keras_layer±
{"class_name": "QuantizeWrapper", "name": "quant_Conv2D-3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "quant_Conv2D-3", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "Conv2D-3", "trainable": true, "dtype": "float32", "filters": 38, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": {"class_name": "NoOpActivation", "config": {}}}}, "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 6, 38]}}
χ	
	flayer
goptimizer_step
h_weight_vars
i_quantize_activations
j_output_quantizers
k	variables
lregularization_losses
mtrainable_variables
n	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layerθ{"class_name": "QuantizeWrapper", "name": "quant_Btch_Norm-3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "quant_Btch_Norm-3", "trainable": true, "dtype": "float32", "layer": {"class_name": "BatchNormalization", "config": {"name": "Btch_Norm-3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, "quantize_config": {"class_name": "NoOpQuantizeConfig", "config": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 4, 38]}}
λ
	olayer
poptimizer_step
q_weight_vars
r_quantize_activations
s_output_quantizers
t
output_min
u
output_max
v_output_quantizer_vars
w	variables
xregularization_losses
ytrainable_variables
z	keras_api
+&call_and_return_all_conditional_losses
__call__"Ί
_tf_keras_layer {"class_name": "QuantizeWrapper", "name": "quant_re_lu_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "quant_re_lu_8", "trainable": true, "dtype": "float32", "layer": {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": true}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 4, 38]}}
‘
	{layer
|optimizer_step
}_weight_vars
~_quantize_activations
_output_quantizers

output_min

output_max
_output_quantizer_vars
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"ι
_tf_keras_layerΟ{"class_name": "QuantizeWrapper", "name": "quant_GlobalAveragePooling-Layer", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "quant_GlobalAveragePooling-Layer", "trainable": true, "dtype": "float32", "layer": {"class_name": "GlobalAveragePooling2D", "config": {"name": "GlobalAveragePooling-Layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": true}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 4, 38]}}
?


layer
optimizer_step
_weight_vars

kernel_min

kernel_max
_quantize_activations
post_activation_min
post_activation_max
_output_quantizers
	variables
regularization_losses
trainable_variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"«
_tf_keras_layer{"class_name": "QuantizeWrapper", "name": "quant_Output-Layer", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "quant_Output-Layer", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "Output-Layer", "trainable": true, "dtype": "float32", "units": 8, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "linear"}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 38]}}
Κ
	iter
beta_1
beta_2

decay
learning_rate	mδ	mε	mζ	mη	mθ	 mι	£mκ	€mλ	₯mμ	¨mν	©mξ	vο	vπ	vρ	vς	vσ	 vτ	£vυ	€vφ	₯vχ	¨vψ	©vω"
	optimizer
Ά
0
1
2
3
4
5
6
7
8
9
10
'11
012
413
514
15
<16
>17
?18
19
 20
‘21
’22
G23
P24
T25
U26
£27
\28
^29
_30
€31
₯32
¦33
§34
g35
p36
t37
u38
|39
40
41
¨42
©43
44
45
46
47
48"
trackable_list_wrapper
 "
trackable_list_wrapper
y
0
1
2
3
4
 5
£6
€7
₯8
¨9
©10"
trackable_list_wrapper
Σ
ͺlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
?non_trainable_variables
	variables
regularization_losses
trainable_variables
ϋ__call__
ό_default_save_signature
+ϊ&call_and_return_all_conditional_losses
'ϊ"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
):' 2!quantize_layer/quantize_layer_min
):' 2!quantize_layer/quantize_layer_max
:
min_var
max_var"
trackable_dict_wrapper
%:# 2quantize_layer/optimizer_step
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
―layers
°metrics
 ±layer_regularization_losses
	variables
²non_trainable_variables
³layer_metrics
regularization_losses
trainable_variables
ώ__call__
+ύ&call_and_return_all_conditional_losses
'ύ"call_and_return_conditional_losses"
_generic_user_object
ά

kernel
΄	variables
΅regularization_losses
Άtrainable_variables
·	keras_api
+&call_and_return_all_conditional_losses
__call__"Ί	
_tf_keras_layer 	{"class_name": "Conv2D", "name": "Conv2D-1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv2D-1", "trainable": true, "dtype": "float32", "filters": 38, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": {"class_name": "NoOpActivation", "config": {}}}}, "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49, 10, 1]}}
%:# 2quant_Conv2D-1/optimizer_step
(
Έ0"
trackable_list_wrapper
%:#&2quant_Conv2D-1/kernel_min
%:#&2quant_Conv2D-1/kernel_max
 "
trackable_list_wrapper
 "
trackable_list_wrapper
=
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
΅
Ήlayers
Ίmetrics
 »layer_regularization_losses
"	variables
Όnon_trainable_variables
½layer_metrics
#regularization_losses
$trainable_variables
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
―	
	Ύaxis

gamma
	beta
moving_mean
moving_variance
Ώ	variables
ΐregularization_losses
Αtrainable_variables
Β	keras_api
+&call_and_return_all_conditional_losses
__call__"Π
_tf_keras_layerΆ{"class_name": "BatchNormalization", "name": "Btch_Norm-1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Btch_Norm-1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 38}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 8, 38]}}
(:& 2 quant_Btch_Norm-1/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
G
0
1
2
3
'4"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
΅
Γlayers
Δmetrics
 Εlayer_regularization_losses
+	variables
Ζnon_trainable_variables
Ηlayer_metrics
,regularization_losses
-trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ρ
Θ	variables
Ιregularization_losses
Κtrainable_variables
Λ	keras_api
+&call_and_return_all_conditional_losses
__call__"ά
_tf_keras_layerΒ{"class_name": "ReLU", "name": "re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
$:" 2quant_re_lu_6/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 : 2quant_re_lu_6/output_min
 : 2quant_re_lu_6/output_max
:
4min_var
5max_var"
trackable_dict_wrapper
5
00
41
52"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
Μlayers
Νmetrics
 Ξlayer_regularization_losses
7	variables
Οnon_trainable_variables
Πlayer_metrics
8regularization_losses
9trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
έ

kernel
Ρ	variables
?regularization_losses
Σtrainable_variables
Τ	keras_api
+&call_and_return_all_conditional_losses
__call__"»	
_tf_keras_layer‘	{"class_name": "Conv2D", "name": "Conv2D-2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv2D-2", "trainable": true, "dtype": "float32", "filters": 38, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": {"class_name": "NoOpActivation", "config": {}}}}, "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 38}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 8, 38]}}
%:# 2quant_Conv2D-2/optimizer_step
(
Υ0"
trackable_list_wrapper
%:#&2quant_Conv2D-2/kernel_min
%:#&2quant_Conv2D-2/kernel_max
 "
trackable_list_wrapper
 "
trackable_list_wrapper
=
0
<1
>2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
΅
Φlayers
Χmetrics
 Ψlayer_regularization_losses
B	variables
Ωnon_trainable_variables
Ϊlayer_metrics
Cregularization_losses
Dtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
―	
	Ϋaxis

gamma
	 beta
‘moving_mean
’moving_variance
ά	variables
έregularization_losses
ήtrainable_variables
ί	keras_api
+&call_and_return_all_conditional_losses
__call__"Π
_tf_keras_layerΆ{"class_name": "BatchNormalization", "name": "Btch_Norm-2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Btch_Norm-2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 38}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 6, 38]}}
(:& 2 quant_Btch_Norm-2/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
G
0
 1
‘2
’3
G4"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
 1"
trackable_list_wrapper
΅
ΰlayers
αmetrics
 βlayer_regularization_losses
K	variables
γnon_trainable_variables
δlayer_metrics
Lregularization_losses
Mtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ρ
ε	variables
ζregularization_losses
ηtrainable_variables
θ	keras_api
+ &call_and_return_all_conditional_losses
‘__call__"ά
_tf_keras_layerΒ{"class_name": "ReLU", "name": "re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
$:" 2quant_re_lu_7/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 : 2quant_re_lu_7/output_min
 : 2quant_re_lu_7/output_max
:
Tmin_var
Umax_var"
trackable_dict_wrapper
5
P0
T1
U2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
ιlayers
κmetrics
 λlayer_regularization_losses
W	variables
μnon_trainable_variables
νlayer_metrics
Xregularization_losses
Ytrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
έ

£kernel
ξ	variables
οregularization_losses
πtrainable_variables
ρ	keras_api
+’&call_and_return_all_conditional_losses
£__call__"»	
_tf_keras_layer‘	{"class_name": "Conv2D", "name": "Conv2D-3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Conv2D-3", "trainable": true, "dtype": "float32", "filters": 38, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": {"class_name": "NoOpActivation", "config": {}}}}, "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 38}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 22, 6, 38]}}
%:# 2quant_Conv2D-3/optimizer_step
(
ς0"
trackable_list_wrapper
%:#&2quant_Conv2D-3/kernel_min
%:#&2quant_Conv2D-3/kernel_max
 "
trackable_list_wrapper
 "
trackable_list_wrapper
=
£0
\1
^2
_3"
trackable_list_wrapper
 "
trackable_list_wrapper
(
£0"
trackable_list_wrapper
΅
σlayers
τmetrics
 υlayer_regularization_losses
b	variables
φnon_trainable_variables
χlayer_metrics
cregularization_losses
dtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
―	
	ψaxis

€gamma
	₯beta
¦moving_mean
§moving_variance
ω	variables
ϊregularization_losses
ϋtrainable_variables
ό	keras_api
+€&call_and_return_all_conditional_losses
₯__call__"Π
_tf_keras_layerΆ{"class_name": "BatchNormalization", "name": "Btch_Norm-3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Btch_Norm-3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.1, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 38}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20, 4, 38]}}
(:& 2 quant_Btch_Norm-3/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
G
€0
₯1
¦2
§3
g4"
trackable_list_wrapper
 "
trackable_list_wrapper
0
€0
₯1"
trackable_list_wrapper
΅
ύlayers
ώmetrics
 ?layer_regularization_losses
k	variables
non_trainable_variables
layer_metrics
lregularization_losses
mtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ρ
	variables
regularization_losses
trainable_variables
	keras_api
+¦&call_and_return_all_conditional_losses
§__call__"ά
_tf_keras_layerΒ{"class_name": "ReLU", "name": "re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": null, "negative_slope": 0.0, "threshold": 0.0}}
$:" 2quant_re_lu_8/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 : 2quant_re_lu_8/output_min
 : 2quant_re_lu_8/output_max
:
tmin_var
umax_var"
trackable_dict_wrapper
5
p0
t1
u2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
layers
metrics
 layer_regularization_losses
w	variables
non_trainable_variables
layer_metrics
xregularization_losses
ytrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object

	variables
regularization_losses
trainable_variables
	keras_api
+¨&call_and_return_all_conditional_losses
©__call__"
_tf_keras_layerξ{"class_name": "GlobalAveragePooling2D", "name": "GlobalAveragePooling-Layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "GlobalAveragePooling-Layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
7:5 2/quant_GlobalAveragePooling-Layer/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
3:1 2+quant_GlobalAveragePooling-Layer/output_min
3:1 2+quant_GlobalAveragePooling-Layer/output_max
<
min_var
max_var"
trackable_dict_wrapper
7
|0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
layers
metrics
 layer_regularization_losses
	variables
non_trainable_variables
layer_metrics
regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Θ
©kernel
	¨bias
	variables
regularization_losses
trainable_variables
	keras_api
+ͺ&call_and_return_all_conditional_losses
«__call__"
_tf_keras_layer{"class_name": "Dense", "name": "Output-Layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "Output-Layer", "trainable": true, "dtype": "float32", "units": 8, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "linear"}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 38}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 38]}}
):' 2!quant_Output-Layer/optimizer_step
(
0"
trackable_list_wrapper
%:# 2quant_Output-Layer/kernel_min
%:# 2quant_Output-Layer/kernel_max
 "
trackable_list_wrapper
.:, 2&quant_Output-Layer/post_activation_min
.:, 2&quant_Output-Layer/post_activation_max
 "
trackable_list_wrapper
X
¨0
©1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
0
¨0
©1"
trackable_list_wrapper
Έ
layers
metrics
 layer_regularization_losses
	variables
non_trainable_variables
layer_metrics
regularization_losses
trainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'&2Conv2D-1/kernel
:&2Btch_Norm-1/gamma
:&2Btch_Norm-1/beta
':%& (2Btch_Norm-1/moving_mean
+:)& (2Btch_Norm-1/moving_variance
):'&&2Conv2D-2/kernel
:&2Btch_Norm-2/gamma
:&2Btch_Norm-2/beta
':%& (2Btch_Norm-2/moving_mean
+:)& (2Btch_Norm-2/moving_variance
):'&&2Conv2D-3/kernel
:&2Btch_Norm-3/gamma
:&2Btch_Norm-3/beta
':%& (2Btch_Norm-3/moving_mean
+:)& (2Btch_Norm-3/moving_variance
:2Output-Layer/bias
%:#&2Output-Layer/kernel
v
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
11"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Σ
0
1
2
3
4
5
6
7
'8
09
410
511
<12
>13
?14
‘15
’16
G17
P18
T19
U20
\21
^22
_23
¦24
§25
g26
p27
t28
u29
|30
31
32
33
34
35
36
37"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 layers
‘metrics
 ’layer_regularization_losses
΄	variables
£non_trainable_variables
€layer_metrics
΅regularization_losses
Άtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
1
0
₯2"
trackable_tuple_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
Έ
¦layers
§metrics
 ¨layer_regularization_losses
Ώ	variables
©non_trainable_variables
ͺlayer_metrics
ΐregularization_losses
Αtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
'
&0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
7
0
1
'2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
«layers
¬metrics
 ­layer_regularization_losses
Θ	variables
?non_trainable_variables
―layer_metrics
Ιregularization_losses
Κtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
00
41
52"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
°layers
±metrics
 ²layer_regularization_losses
Ρ	variables
³non_trainable_variables
΄layer_metrics
?regularization_losses
Σtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
1
0
΅2"
trackable_tuple_wrapper
'
;0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
<0
>1
?2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
@
0
 1
‘2
’3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
 1"
trackable_list_wrapper
Έ
Άlayers
·metrics
 Έlayer_regularization_losses
ά	variables
Ήnon_trainable_variables
Ίlayer_metrics
έregularization_losses
ήtrainable_variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
'
F0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
7
‘0
’1
G2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
»layers
Όmetrics
 ½layer_regularization_losses
ε	variables
Ύnon_trainable_variables
Ώlayer_metrics
ζregularization_losses
ηtrainable_variables
‘__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
'
O0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
P0
T1
U2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ΐlayers
Αmetrics
 Βlayer_regularization_losses
ξ	variables
Γnon_trainable_variables
Δlayer_metrics
οregularization_losses
πtrainable_variables
£__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses"
_generic_user_object
1
£0
Ε2"
trackable_tuple_wrapper
'
[0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
\0
^1
_2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
@
€0
₯1
¦2
§3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
€0
₯1"
trackable_list_wrapper
Έ
Ζlayers
Ηmetrics
 Θlayer_regularization_losses
ω	variables
Ιnon_trainable_variables
Κlayer_metrics
ϊregularization_losses
ϋtrainable_variables
₯__call__
+€&call_and_return_all_conditional_losses
'€"call_and_return_conditional_losses"
_generic_user_object
'
f0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
7
¦0
§1
g2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Λlayers
Μmetrics
 Νlayer_regularization_losses
	variables
Ξnon_trainable_variables
Οlayer_metrics
regularization_losses
trainable_variables
§__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
'
o0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
p0
t1
u2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Πlayers
Ρmetrics
 ?layer_regularization_losses
	variables
Σnon_trainable_variables
Τlayer_metrics
regularization_losses
trainable_variables
©__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
'
{0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
7
|0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
(
¨0"
trackable_list_wrapper
 "
trackable_list_wrapper
(
¨0"
trackable_list_wrapper
Έ
Υlayers
Φmetrics
 Χlayer_regularization_losses
	variables
Ψnon_trainable_variables
Ωlayer_metrics
regularization_losses
trainable_variables
«__call__
+ͺ&call_and_return_all_conditional_losses
'ͺ"call_and_return_conditional_losses"
_generic_user_object
1
©0
Ϊ2"
trackable_tuple_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
H
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
Ώ

Ϋtotal

άcount
έ	variables
ή	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


ίtotal

ΰcount
α
_fn_kwargs
β	variables
γ	keras_api"Θ
_tf_keras_metric­{"class_name": "SparseCategoricalAccuracy", "name": "sparse_categorical_accuracy", "dtype": "float32", "config": {"name": "sparse_categorical_accuracy", "dtype": "float32"}}
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
:
min_var
max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
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
:
>min_var
?max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
‘0
’1"
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
:
^min_var
_max_var"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
¦0
§1"
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
<
min_var
max_var"
trackable_dict_wrapper
:  (2total
:  (2count
0
Ϋ0
ά1"
trackable_list_wrapper
.
έ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ί0
ΰ1"
trackable_list_wrapper
.
β	variables"
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
$:"2Adam/Output-Layer/bias/m
*:(&2Adam/Output-Layer/kernel/m
.:,&2Adam/Conv2D-1/kernel/v
$:"&2Adam/Btch_Norm-1/gamma/v
#:!&2Adam/Btch_Norm-1/beta/v
.:,&&2Adam/Conv2D-2/kernel/v
$:"&2Adam/Btch_Norm-2/gamma/v
#:!&2Adam/Btch_Norm-2/beta/v
.:,&&2Adam/Conv2D-3/kernel/v
$:"&2Adam/Btch_Norm-3/gamma/v
#:!&2Adam/Btch_Norm-3/beta/v
$:"2Adam/Output-Layer/bias/v
*:(&2Adam/Output-Layer/kernel/v
κ2η
G__inference_sequential_4_layer_call_and_return_conditional_losses_59154
G__inference_sequential_4_layer_call_and_return_conditional_losses_58340
G__inference_sequential_4_layer_call_and_return_conditional_losses_58250
G__inference_sequential_4_layer_call_and_return_conditional_losses_59038ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ώ2ϋ
,__inference_sequential_4_layer_call_fn_59233
,__inference_sequential_4_layer_call_fn_58510
,__inference_sequential_4_layer_call_fn_59312
,__inference_sequential_4_layer_call_fn_58679ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ν2κ
 __inference__wrapped_model_57068Ε
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *5’2
0-
Conv2D-1_input?????????1

Π2Ν
I__inference_quantize_layer_layer_call_and_return_conditional_losses_59333
I__inference_quantize_layer_layer_call_and_return_conditional_losses_59342΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
.__inference_quantize_layer_layer_call_fn_59360
.__inference_quantize_layer_layer_call_fn_59351΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Π2Ν
I__inference_quant_Conv2D-1_layer_call_and_return_conditional_losses_59384
I__inference_quant_Conv2D-1_layer_call_and_return_conditional_losses_59396΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
.__inference_quant_Conv2D-1_layer_call_fn_59407
.__inference_quant_Conv2D-1_layer_call_fn_59418΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Φ2Σ
L__inference_quant_Btch_Norm-1_layer_call_and_return_conditional_losses_59438
L__inference_quant_Btch_Norm-1_layer_call_and_return_conditional_losses_59456΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 2
1__inference_quant_Btch_Norm-1_layer_call_fn_59469
1__inference_quant_Btch_Norm-1_layer_call_fn_59482΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ξ2Λ
H__inference_quant_re_lu_6_layer_call_and_return_conditional_losses_59508
H__inference_quant_re_lu_6_layer_call_and_return_conditional_losses_59518΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
-__inference_quant_re_lu_6_layer_call_fn_59536
-__inference_quant_re_lu_6_layer_call_fn_59527΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Π2Ν
I__inference_quant_Conv2D-2_layer_call_and_return_conditional_losses_59560
I__inference_quant_Conv2D-2_layer_call_and_return_conditional_losses_59572΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
.__inference_quant_Conv2D-2_layer_call_fn_59594
.__inference_quant_Conv2D-2_layer_call_fn_59583΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Φ2Σ
L__inference_quant_Btch_Norm-2_layer_call_and_return_conditional_losses_59614
L__inference_quant_Btch_Norm-2_layer_call_and_return_conditional_losses_59632΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 2
1__inference_quant_Btch_Norm-2_layer_call_fn_59645
1__inference_quant_Btch_Norm-2_layer_call_fn_59658΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ξ2Λ
H__inference_quant_re_lu_7_layer_call_and_return_conditional_losses_59694
H__inference_quant_re_lu_7_layer_call_and_return_conditional_losses_59684΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
-__inference_quant_re_lu_7_layer_call_fn_59712
-__inference_quant_re_lu_7_layer_call_fn_59703΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Π2Ν
I__inference_quant_Conv2D-3_layer_call_and_return_conditional_losses_59748
I__inference_quant_Conv2D-3_layer_call_and_return_conditional_losses_59736΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
.__inference_quant_Conv2D-3_layer_call_fn_59759
.__inference_quant_Conv2D-3_layer_call_fn_59770΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Φ2Σ
L__inference_quant_Btch_Norm-3_layer_call_and_return_conditional_losses_59808
L__inference_quant_Btch_Norm-3_layer_call_and_return_conditional_losses_59790΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 2
1__inference_quant_Btch_Norm-3_layer_call_fn_59821
1__inference_quant_Btch_Norm-3_layer_call_fn_59834΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ξ2Λ
H__inference_quant_re_lu_8_layer_call_and_return_conditional_losses_59860
H__inference_quant_re_lu_8_layer_call_and_return_conditional_losses_59870΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
-__inference_quant_re_lu_8_layer_call_fn_59879
-__inference_quant_re_lu_8_layer_call_fn_59888΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
τ2ρ
[__inference_quant_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_59915
[__inference_quant_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_59926΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ύ2»
@__inference_quant_GlobalAveragePooling-Layer_layer_call_fn_59935
@__inference_quant_GlobalAveragePooling-Layer_layer_call_fn_59944΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ψ2Υ
M__inference_quant_Output-Layer_layer_call_and_return_conditional_losses_60020
M__inference_quant_Output-Layer_layer_call_and_return_conditional_losses_60000΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
’2
2__inference_quant_Output-Layer_layer_call_fn_60054
2__inference_quant_Output-Layer_layer_call_fn_60037΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
9B7
#__inference_signature_wrapper_58768Conv2D-1_input
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Κ2Η
F__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_60092
F__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_60074΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
+__inference_Btch_Norm-1_layer_call_fn_60118
+__inference_Btch_Norm-1_layer_call_fn_60105΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Κ2Η
F__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_60156
F__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_60138΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
+__inference_Btch_Norm-2_layer_call_fn_60182
+__inference_Btch_Norm-2_layer_call_fn_60169΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Κ2Η
F__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_60220
F__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_60202΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
+__inference_Btch_Norm-3_layer_call_fn_60233
+__inference_Btch_Norm-3_layer_call_fn_60246΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
½2Ί
U__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_57387ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
’2
:__inference_GlobalAveragePooling-Layer_layer_call_fn_57393ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
¨2₯’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ε
F__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_60074M’J
C’@
:7
inputs+???????????????????????????&
p
ͺ "?’<
52
0+???????????????????????????&
 ε
F__inference_Btch_Norm-1_layer_call_and_return_conditional_losses_60092M’J
C’@
:7
inputs+???????????????????????????&
p 
ͺ "?’<
52
0+???????????????????????????&
 ½
+__inference_Btch_Norm-1_layer_call_fn_60105M’J
C’@
:7
inputs+???????????????????????????&
p
ͺ "2/+???????????????????????????&½
+__inference_Btch_Norm-1_layer_call_fn_60118M’J
C’@
:7
inputs+???????????????????????????&
p 
ͺ "2/+???????????????????????????&ε
F__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_60138 ‘’M’J
C’@
:7
inputs+???????????????????????????&
p
ͺ "?’<
52
0+???????????????????????????&
 ε
F__inference_Btch_Norm-2_layer_call_and_return_conditional_losses_60156 ‘’M’J
C’@
:7
inputs+???????????????????????????&
p 
ͺ "?’<
52
0+???????????????????????????&
 ½
+__inference_Btch_Norm-2_layer_call_fn_60169 ‘’M’J
C’@
:7
inputs+???????????????????????????&
p
ͺ "2/+???????????????????????????&½
+__inference_Btch_Norm-2_layer_call_fn_60182 ‘’M’J
C’@
:7
inputs+???????????????????????????&
p 
ͺ "2/+???????????????????????????&ε
F__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_60202€₯¦§M’J
C’@
:7
inputs+???????????????????????????&
p
ͺ "?’<
52
0+???????????????????????????&
 ε
F__inference_Btch_Norm-3_layer_call_and_return_conditional_losses_60220€₯¦§M’J
C’@
:7
inputs+???????????????????????????&
p 
ͺ "?’<
52
0+???????????????????????????&
 ½
+__inference_Btch_Norm-3_layer_call_fn_60233€₯¦§M’J
C’@
:7
inputs+???????????????????????????&
p
ͺ "2/+???????????????????????????&½
+__inference_Btch_Norm-3_layer_call_fn_60246€₯¦§M’J
C’@
:7
inputs+???????????????????????????&
p 
ͺ "2/+???????????????????????????&ή
U__inference_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_57387R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ".’+
$!
0??????????????????
 ΅
:__inference_GlobalAveragePooling-Layer_layer_call_fn_57393wR’O
H’E
C@
inputs4????????????????????????????????????
ͺ "!??????????????????ν
 __inference__wrapped_model_57068Θ<45>? ‘’TU£^_€₯¦§tu©¨?’<
5’2
0-
Conv2D-1_input?????????1

ͺ "GͺD
B
quant_Output-Layer,)
quant_Output-Layer?????????Ζ
L__inference_quant_Btch_Norm-1_layer_call_and_return_conditional_losses_59438v;’8
1’.
(%
inputs?????????&
p
ͺ "-’*
# 
0?????????&
 Ζ
L__inference_quant_Btch_Norm-1_layer_call_and_return_conditional_losses_59456v;’8
1’.
(%
inputs?????????&
p 
ͺ "-’*
# 
0?????????&
 
1__inference_quant_Btch_Norm-1_layer_call_fn_59469i;’8
1’.
(%
inputs?????????&
p
ͺ " ?????????&
1__inference_quant_Btch_Norm-1_layer_call_fn_59482i;’8
1’.
(%
inputs?????????&
p 
ͺ " ?????????&Ζ
L__inference_quant_Btch_Norm-2_layer_call_and_return_conditional_losses_59614v ‘’;’8
1’.
(%
inputs?????????&
p
ͺ "-’*
# 
0?????????&
 Ζ
L__inference_quant_Btch_Norm-2_layer_call_and_return_conditional_losses_59632v ‘’;’8
1’.
(%
inputs?????????&
p 
ͺ "-’*
# 
0?????????&
 
1__inference_quant_Btch_Norm-2_layer_call_fn_59645i ‘’;’8
1’.
(%
inputs?????????&
p
ͺ " ?????????&
1__inference_quant_Btch_Norm-2_layer_call_fn_59658i ‘’;’8
1’.
(%
inputs?????????&
p 
ͺ " ?????????&Ζ
L__inference_quant_Btch_Norm-3_layer_call_and_return_conditional_losses_59790v€₯¦§;’8
1’.
(%
inputs?????????&
p
ͺ "-’*
# 
0?????????&
 Ζ
L__inference_quant_Btch_Norm-3_layer_call_and_return_conditional_losses_59808v€₯¦§;’8
1’.
(%
inputs?????????&
p 
ͺ "-’*
# 
0?????????&
 
1__inference_quant_Btch_Norm-3_layer_call_fn_59821i€₯¦§;’8
1’.
(%
inputs?????????&
p
ͺ " ?????????&
1__inference_quant_Btch_Norm-3_layer_call_fn_59834i€₯¦§;’8
1’.
(%
inputs?????????&
p 
ͺ " ?????????&Ώ
I__inference_quant_Conv2D-1_layer_call_and_return_conditional_losses_59384r;’8
1’.
(%
inputs?????????1

p
ͺ "-’*
# 
0?????????&
 Ώ
I__inference_quant_Conv2D-1_layer_call_and_return_conditional_losses_59396r;’8
1’.
(%
inputs?????????1

p 
ͺ "-’*
# 
0?????????&
 
.__inference_quant_Conv2D-1_layer_call_fn_59407e;’8
1’.
(%
inputs?????????1

p
ͺ " ?????????&
.__inference_quant_Conv2D-1_layer_call_fn_59418e;’8
1’.
(%
inputs?????????1

p 
ͺ " ?????????&Ώ
I__inference_quant_Conv2D-2_layer_call_and_return_conditional_losses_59560r>?;’8
1’.
(%
inputs?????????&
p
ͺ "-’*
# 
0?????????&
 Ώ
I__inference_quant_Conv2D-2_layer_call_and_return_conditional_losses_59572r>?;’8
1’.
(%
inputs?????????&
p 
ͺ "-’*
# 
0?????????&
 
.__inference_quant_Conv2D-2_layer_call_fn_59583e>?;’8
1’.
(%
inputs?????????&
p
ͺ " ?????????&
.__inference_quant_Conv2D-2_layer_call_fn_59594e>?;’8
1’.
(%
inputs?????????&
p 
ͺ " ?????????&Ώ
I__inference_quant_Conv2D-3_layer_call_and_return_conditional_losses_59736r£^_;’8
1’.
(%
inputs?????????&
p
ͺ "-’*
# 
0?????????&
 Ώ
I__inference_quant_Conv2D-3_layer_call_and_return_conditional_losses_59748r£^_;’8
1’.
(%
inputs?????????&
p 
ͺ "-’*
# 
0?????????&
 
.__inference_quant_Conv2D-3_layer_call_fn_59759e£^_;’8
1’.
(%
inputs?????????&
p
ͺ " ?????????&
.__inference_quant_Conv2D-3_layer_call_fn_59770e£^_;’8
1’.
(%
inputs?????????&
p 
ͺ " ?????????&Ι
[__inference_quant_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_59915j;’8
1’.
(%
inputs?????????&
p
ͺ "%’"

0?????????&
 Ι
[__inference_quant_GlobalAveragePooling-Layer_layer_call_and_return_conditional_losses_59926j;’8
1’.
(%
inputs?????????&
p 
ͺ "%’"

0?????????&
 ‘
@__inference_quant_GlobalAveragePooling-Layer_layer_call_fn_59935];’8
1’.
(%
inputs?????????&
p
ͺ "?????????&‘
@__inference_quant_GlobalAveragePooling-Layer_layer_call_fn_59944];’8
1’.
(%
inputs?????????&
p 
ͺ "?????????&»
M__inference_quant_Output-Layer_layer_call_and_return_conditional_losses_60000j©¨3’0
)’&
 
inputs?????????&
p
ͺ "%’"

0?????????
 »
M__inference_quant_Output-Layer_layer_call_and_return_conditional_losses_60020j©¨3’0
)’&
 
inputs?????????&
p 
ͺ "%’"

0?????????
 
2__inference_quant_Output-Layer_layer_call_fn_60037]©¨3’0
)’&
 
inputs?????????&
p
ͺ "?????????
2__inference_quant_Output-Layer_layer_call_fn_60054]©¨3’0
)’&
 
inputs?????????&
p 
ͺ "?????????Ό
H__inference_quant_re_lu_6_layer_call_and_return_conditional_losses_59508p45;’8
1’.
(%
inputs?????????&
p
ͺ "-’*
# 
0?????????&
 Ό
H__inference_quant_re_lu_6_layer_call_and_return_conditional_losses_59518p45;’8
1’.
(%
inputs?????????&
p 
ͺ "-’*
# 
0?????????&
 
-__inference_quant_re_lu_6_layer_call_fn_59527c45;’8
1’.
(%
inputs?????????&
p
ͺ " ?????????&
-__inference_quant_re_lu_6_layer_call_fn_59536c45;’8
1’.
(%
inputs?????????&
p 
ͺ " ?????????&Ό
H__inference_quant_re_lu_7_layer_call_and_return_conditional_losses_59684pTU;’8
1’.
(%
inputs?????????&
p
ͺ "-’*
# 
0?????????&
 Ό
H__inference_quant_re_lu_7_layer_call_and_return_conditional_losses_59694pTU;’8
1’.
(%
inputs?????????&
p 
ͺ "-’*
# 
0?????????&
 
-__inference_quant_re_lu_7_layer_call_fn_59703cTU;’8
1’.
(%
inputs?????????&
p
ͺ " ?????????&
-__inference_quant_re_lu_7_layer_call_fn_59712cTU;’8
1’.
(%
inputs?????????&
p 
ͺ " ?????????&Ό
H__inference_quant_re_lu_8_layer_call_and_return_conditional_losses_59860ptu;’8
1’.
(%
inputs?????????&
p
ͺ "-’*
# 
0?????????&
 Ό
H__inference_quant_re_lu_8_layer_call_and_return_conditional_losses_59870ptu;’8
1’.
(%
inputs?????????&
p 
ͺ "-’*
# 
0?????????&
 
-__inference_quant_re_lu_8_layer_call_fn_59879ctu;’8
1’.
(%
inputs?????????&
p
ͺ " ?????????&
-__inference_quant_re_lu_8_layer_call_fn_59888ctu;’8
1’.
(%
inputs?????????&
p 
ͺ " ?????????&½
I__inference_quantize_layer_layer_call_and_return_conditional_losses_59333p;’8
1’.
(%
inputs?????????1

p
ͺ "-’*
# 
0?????????1

 ½
I__inference_quantize_layer_layer_call_and_return_conditional_losses_59342p;’8
1’.
(%
inputs?????????1

p 
ͺ "-’*
# 
0?????????1

 
.__inference_quantize_layer_layer_call_fn_59351c;’8
1’.
(%
inputs?????????1

p
ͺ " ?????????1

.__inference_quantize_layer_layer_call_fn_59360c;’8
1’.
(%
inputs?????????1

p 
ͺ " ?????????1
ϊ
G__inference_sequential_4_layer_call_and_return_conditional_losses_58250?<45>? ‘’TU£^_€₯¦§tu©¨G’D
=’:
0-
Conv2D-1_input?????????1

p

 
ͺ "%’"

0?????????
 ϊ
G__inference_sequential_4_layer_call_and_return_conditional_losses_58340?<45>? ‘’TU£^_€₯¦§tu©¨G’D
=’:
0-
Conv2D-1_input?????????1

p 

 
ͺ "%’"

0?????????
 ς
G__inference_sequential_4_layer_call_and_return_conditional_losses_59038¦<45>? ‘’TU£^_€₯¦§tu©¨?’<
5’2
(%
inputs?????????1

p

 
ͺ "%’"

0?????????
 ς
G__inference_sequential_4_layer_call_and_return_conditional_losses_59154¦<45>? ‘’TU£^_€₯¦§tu©¨?’<
5’2
(%
inputs?????????1

p 

 
ͺ "%’"

0?????????
 ?
,__inference_sequential_4_layer_call_fn_58510‘<45>? ‘’TU£^_€₯¦§tu©¨G’D
=’:
0-
Conv2D-1_input?????????1

p

 
ͺ "??????????
,__inference_sequential_4_layer_call_fn_58679‘<45>? ‘’TU£^_€₯¦§tu©¨G’D
=’:
0-
Conv2D-1_input?????????1

p 

 
ͺ "?????????Κ
,__inference_sequential_4_layer_call_fn_59233<45>? ‘’TU£^_€₯¦§tu©¨?’<
5’2
(%
inputs?????????1

p

 
ͺ "?????????Κ
,__inference_sequential_4_layer_call_fn_59312<45>? ‘’TU£^_€₯¦§tu©¨?’<
5’2
(%
inputs?????????1

p 

 
ͺ "?????????
#__inference_signature_wrapper_58768Ϊ<45>? ‘’TU£^_€₯¦§tu©¨Q’N
’ 
GͺD
B
Conv2D-1_input0-
Conv2D-1_input?????????1
"GͺD
B
quant_Output-Layer,)
quant_Output-Layer?????????