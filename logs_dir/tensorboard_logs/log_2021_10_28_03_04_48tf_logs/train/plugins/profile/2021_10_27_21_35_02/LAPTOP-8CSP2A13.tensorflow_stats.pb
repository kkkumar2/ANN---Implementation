"�L
BHostIDLE"IDLE1?5^��&�@A?5^��&�@a�����?i�����?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1;�O��g�@9;�O��g�@A;�O��g�@I;�O��g�@ay�\4��?i�����?�Unknown�
vHost_FusedMatMul"sequential/hiddenlayer1/Relu(1��C�l��@9��C�l��@A��C�l��@I��C�l��@a>Y}L��?i��t$���?�Unknown
xHost_FusedMatMul"sequential/outputlayer/BiasAdd(1�x�&1o@9�x�&1o@A�x�&1o@I�x�&1o@aFb�R2��?in����?�Unknown
�HostMatMul",gradient_tape/sequential/hiddenlayer1/MatMul(1ףp=
�n@9ףp=
�n@Aףp=
�n@Iףp=
�n@aG�i#q�?i4�g{[G�?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1�|?5^*m@9�|?5^*m@A�|?5^*m@I�|?5^*m@a���`?�?i��xY��?�Unknown
vHost_FusedMatMul"sequential/hiddenlayer2/Relu(1�(\��i@9�(\��i@A�(\��i@I�(\��i@a�Ͽ��?i2�N����?�Unknown
�HostMatMul",gradient_tape/sequential/hiddenlayer2/MatMul(1�|?5^.e@9�|?5^.e@A�|?5^.e@I�|?5^.e@aC�J(�?i:�	y-�?�Unknown
�	HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1�p=
׿d@9�p=
׿d@A�p=
׿d@I�p=
׿d@aP迟��?i%S��l�?�Unknown
�
HostMatMul".gradient_tape/sequential/hiddenlayer2/MatMul_1(1^�I�`@9^�I�`@A^�I�`@I^�I�`@a["�	4Cy?iP5f\O��?�Unknown
HostMatMul"+gradient_tape/sequential/outputlayer/MatMul(1y�&1,M@9y�&1,M@Ay�&1,M@Iy�&1,M@a�Z�S�@f?i�"� ���?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1X9��v�L@9X9��v�L@AX9��v�L@IX9��v�L@a�#-.d�e?i�O脉��?�Unknown
iHostWriteSummary"WriteSummary(1=
ףp]J@9=
ףp]J@A=
ףp]J@I=
ףp]J@a,)<�d?i�x$!���?�Unknown�
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1�E����G@9�E����G@A�E����G@I�E����G@aR�Pq��a?i�ɕ���?�Unknown
sHostSoftmax"sequential/outputlayer/Softmax(1��ʡEG@9��ʡEG@A��ʡEG@I��ʡEG@a$mLNE�a?i&�U'�?�Unknown
^HostGatherV2"GatherV2(1L7�A`eF@9L7�A`eF@AL7�A`eF@IL7�A`eF@a�6�a?i�<�?�Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1�Zd;E@9�Zd;E@A�Zd;E@I�Zd;E@aMÝK:2`?iеeo$�?�Unknown
�HostMatMul"-gradient_tape/sequential/outputlayer/MatMul_1(1+���D@9+���D@A+���D@I+���D@aN�;D�_?iw���Z4�?�Unknown
�HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1��"��H@9��"��H@AZd;�O=D@IZd;�O=D@a��'��^?iG)�$�C�?�Unknown
�HostReluGrad".gradient_tape/sequential/hiddenlayer1/ReluGrad(1㥛� �:@9㥛� �:@A㥛� �:@I㥛� �:@a���tT?i���0N�?�Unknown
cHostDataset"Iterator::Root(1����M�P@9����M�P@A�O��n�8@I�O��n�8@a�t?M�S?idS�
�W�?�Unknown
�HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1P��nsA@9P��nsA@A�|?5^z7@I�|?5^z7@a;�:���Q?i�p�t}`�?�Unknown
�HostBiasAddGrad"8gradient_tape/sequential/outputlayer/BiasAdd/BiasAddGrad(1���S��3@9���S��3@A���S��3@I���S��3@adq�iq[N?i��OQh�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1D�l���1@9D�l���1@AD�l���1@ID�l���1@a�a�TK?icGhu�n�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1��Mb81@9��Mb81@A��Mb81@I��Mb81@a#��:EJ?i�6�zu�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1��|?5�0@9��|?5�0@A��|?5�0@I��|?5�0@aӇM��I?irJd��{�?�Unknown
ZHostArgMax"ArgMax(1�C�l��,@9�C�l��,@A�C�l��,@I�C�l��,@a�|��F?i-��p��?�Unknown
gHostStridedSlice"strided_slice(1'1��,@9'1��,@A'1��,@I'1��,@ac���t�E?iw� ��?�Unknown
�HostReluGrad".gradient_tape/sequential/hiddenlayer2/ReluGrad(1q=
ףp+@9q=
ףp+@Aq=
ףp+@Iq=
ףp+@a�^؉�D?i�~� ��?�Unknown
�HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1?5^�I�(@9?5^�I�(@A?5^�I�(@I?5^�I�(@a	�v��B?i]�iې�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1P��n�(@9P��n�(@AP��n�(@IP��n�(@ah*޾��B?i�h�M���?�Unknown
� HostBiasAddGrad"9gradient_tape/sequential/hiddenlayer1/BiasAdd/BiasAddGrad(1��/ݤ(@9��/ݤ(@A��/ݤ(@I��/ݤ(@a9����B?i��oG��?�Unknown
l!HostIteratorGetNext"IteratorGetNext(133333s(@933333s(@A33333s(@I33333s(@aE:.��B?iu���?�Unknown
�"HostDataset"NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1d;�O��&@9d;�O��&@Ad;�O��&@Id;�O��&@ajs캡lA?i�ZqAL��?�Unknown
�#HostReadVariableOp".sequential/hiddenlayer1/BiasAdd/ReadVariableOp(1\���(�&@9\���(�&@A\���(�&@I\���(�&@a�qr�S?A?i.�S���?�Unknown
�$HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1�G�z.&@9�G�z.&@A�G�z.&@I�G�z.&@aAO��Z�@?iB[�֫�?�Unknown
�%HostBiasAddGrad"9gradient_tape/sequential/hiddenlayer2/BiasAdd/BiasAddGrad(1�� �r�%@9�� �r�%@A�� �r�%@I�� �r�%@a�}�j�@?i�z�G���?�Unknown
w&HostDataset""Iterator::Root::ParallelMapV2::Zip(1�(\��UX@9�(\��UX@Au�V%@Iu�V%@a�4A��@?iK�>���?�Unknown
e'Host
LogicalAnd"
LogicalAnd(1��(\��$@9��(\��$@A��(\��$@I��(\��$@a�"D�wJ??i�ӱ���?�Unknown�
v(HostAssignAddVariableOp"AssignAddVariableOp_2(1�(\���#@9�(\���#@A�(\���#@I�(\���#@aX�r�s>?i�����?�Unknown
`)HostGatherV2"
GatherV2_1(1�O��n�"@9�O��n�"@A�O��n�"@I�O��n�"@a<�M�U<?it���>��?�Unknown
V*HostSum"Sum_2(1㥛� 0"@9㥛� 0"@A㥛� 0"@I㥛� 0"@ap���;?i������?�Unknown
w+HostReadVariableOp"SGD/Identity/ReadVariableOp(1sh��|�!@9sh��|�!@Ash��|�!@Ish��|�!@a��_u;?i�2�R%��?�Unknown
|,HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1���Mb� @9���Mb� @A���Mb� @I���Mb� @a��tME9?i����M��?�Unknown
�-HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1�Zd;_@9�Zd;_@A�Zd;_@I�Zd;_@a�Qr(c�7?i����K��?�Unknown
�.HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1Zd;�O�@9Zd;�O�@AZd;�O�@IZd;�O�@a<ÎAN7?iX�.�5��?�Unknown
v/HostCast"$sparse_categorical_crossentropy/Cast(1�Zd�@9�Zd�@A�Zd�@I�Zd�@a����6?i�=-U��?�Unknown
�0HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1��x�&1@9��x�&1@A��x�&1@I��x�&1@a���4?iV9����?�Unknown
t1HostAssignAddVariableOp"AssignAddVariableOp(1T㥛� @9T㥛� @AT㥛� @IT㥛� @a�m�G��4?i�0/L9��?�Unknown
�2HostReadVariableOp"-sequential/hiddenlayer2/MatMul/ReadVariableOp(1�v��/@9�v��/@A�v��/@I�v��/@aU�ɥC�3?i��t���?�Unknown
�3HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1D�l���@9D�l���@AD�l���@ID�l���@a͠��o�2?i8?����?�Unknown
y4HostReadVariableOp"SGD/Identity_1/ReadVariableOp(1����K@9����K@A����K@I����K@a�4�/J�1?i�:�+K��?�Unknown
�5HostReadVariableOp",sequential/outputlayer/MatMul/ReadVariableOp(1y�&1�@9y�&1�@Ay�&1�@Iy�&1�@a��oK1?i��t��?�Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_4(1��S㥛@9��S㥛@A��S㥛@I��S㥛@aPdq��>1?i.ۭw���?�Unknown
X7HostCast"Cast_2(1�����@9�����@A�����@I�����@aSXXSs0?i9F����?�Unknown
u8HostReadVariableOp"div_no_nan/ReadVariableOp(1�I+@9�I+@A�I+@I�I+@a�,Ib7�.?i�j�i���?�Unknown
X9HostEqual"Equal(1ˡE��}@9ˡE��}@AˡE��}@IˡE��}@a`�����*?i��}e3��?�Unknown
�:HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1��K7��@9��K7��@A��K7��@I��K7��@a��tJ�)?i&6%j���?�Unknown
b;HostDivNoNan"div_no_nan_1(1%��C@9%��C@A%��C@I%��C@a���5z(?i�T~T��?�Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_3(1X9��v�@9X9��v�@AX9��v�@IX9��v�@a��5A7(?i3h~���?�Unknown
�=HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1-����@9-����@A-����@I-����@a�8��y'?i�[�O��?�Unknown
�>HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1)\���XC@9)\���XC@A��|?5^@I��|?5^@a�Rb�R*'?i�A/����?�Unknown
�?HostReadVariableOp"-sequential/hiddenlayer1/MatMul/ReadVariableOp(1+���@9+���@A+���@I+���@a��;]��&?i���-��?�Unknown
�@HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1NbX9�@9NbX9�@ANbX9�@INbX9�@a�<��_�%?iŤ���?�Unknown
�AHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1��ʡE@9��ʡE@A��ʡE@I��ʡE@a�Z�%?iXf�����?�Unknown
VBHostCast"Cast(1���x�&@9���x�&@A���x�&@I���x�&@a���I�y%?i��<��?�Unknown
wCHostReadVariableOp"div_no_nan_1/ReadVariableOp(1�(\���@9�(\���@A�(\���@I�(\���@a��o�T%?i��{ӑ��?�Unknown
`DHostDivNoNan"
div_no_nan(1��Q�@9��Q�@A��Q�@I��Q�@a�8�W�$?iA����?�Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_1(1�O��n	@9�O��n	@A�O��n	@I�O��n	@aH1�S #?iN���?�Unknown
�FHostReadVariableOp".sequential/hiddenlayer2/BiasAdd/ReadVariableOp(1�O��n@9�O��n@A�O��n@I�O��n@a\�W�C� ?i�#B:��?�Unknown
TGHostMul"Mul(1o��ʡ@9o��ʡ@Ao��ʡ@Io��ʡ@a�p,W� ?i��?$��?�Unknown
XHHostCast"Cast_3(1�� �rh@9�� �rh@A�� �rh@I�� �rh@a���s�"?i_��T��?�Unknown
�IHostReadVariableOp"-sequential/outputlayer/BiasAdd/ReadVariableOp(1�l����@9�l����@A�l����@I�l����@a�	�M��?i�����?�Unknown
aJHostIdentity"Identity(1��"��~ @9��"��~ @A��"��~ @I��"��~ @aDƃ�*?i9w����?�Unknown�
yKHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1w��/��?9w��/��?Aw��/��?Iw��/��?a_*g/�?ir�:���?�Unknown
XLHostCast"Cast_4(17�A`���?97�A`���?A7�A`���?I7�A`���?a2�Mρ?i�lIF��?�Unknown
wMHostReadVariableOp"div_no_nan/ReadVariableOp_1(1;�O��n�?9;�O��n�?A;�O��n�?I;�O��n�?a���~�6?i      �?�Unknown*�L
uHostFlushSummaryWriter"FlushSummaryWriter(1;�O��g�@9;�O��g�@A;�O��g�@I;�O��g�@aM��f ��?iM��f ��?�Unknown�
vHost_FusedMatMul"sequential/hiddenlayer1/Relu(1��C�l��@9��C�l��@A��C�l��@I��C�l��@an�5��o�?i�?��?�Unknown
xHost_FusedMatMul"sequential/outputlayer/BiasAdd(1�x�&1o@9�x�&1o@A�x�&1o@I�x�&1o@a����W!�?i���(u�?�Unknown
�HostMatMul",gradient_tape/sequential/hiddenlayer1/MatMul(1ףp=
�n@9ףp=
�n@Aףp=
�n@Iףp=
�n@a�C��֘?i��K:�;�?�Unknown
�HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1�|?5^*m@9�|?5^*m@A�|?5^*m@I�|?5^*m@a�x�H=��?i�$k��?�Unknown
vHost_FusedMatMul"sequential/hiddenlayer2/Relu(1�(\��i@9�(\��i@A�(\��i@I�(\��i@a�%��B�?i.�dD���?�Unknown
�HostMatMul",gradient_tape/sequential/hiddenlayer2/MatMul(1�|?5^.e@9�|?5^.e@A�|?5^.e@I�|?5^.e@aw�X�L�?iʶ_�s#�?�Unknown
�HostResourceApplyGradientDescent"+SGD/SGD/update/ResourceApplyGradientDescent(1�p=
׿d@9�p=
׿d@A�p=
׿d@I�p=
׿d@a���Đ?i�_�s���?�Unknown
�	HostMatMul".gradient_tape/sequential/hiddenlayer2/MatMul_1(1^�I�`@9^�I�`@A^�I�`@I^�I�`@a����Ê?i:.PҪ�?�Unknown

HostMatMul"+gradient_tape/sequential/outputlayer/MatMul(1y�&1,M@9y�&1,M@Ay�&1,M@Iy�&1,M@a�򓏕�w?i Vo��C�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_2/ResourceApplyGradientDescent(1X9��v�L@9X9��v�L@AX9��v�L@IX9��v�L@a�����Gw?i����ar�?�Unknown
iHostWriteSummary"WriteSummary(1=
ףp]J@9=
ףp]J@A=
ףp]J@I=
ףp]J@a� g�Nu?i�������?�Unknown�
�Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1�E����G@9�E����G@A�E����G@I�E����G@aJ��s?i�f����?�Unknown
sHostSoftmax"sequential/outputlayer/Softmax(1��ʡEG@9��ʡEG@A��ʡEG@I��ʡEG@aF`�&��r?i��&;��?�Unknown
^HostGatherV2"GatherV2(1L7�A`eF@9L7�A`eF@AL7�A`eF@IL7�A`eF@a)�U�r?iÕ�tn�?�Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1�Zd;E@9�Zd;E@A�Zd;E@I�Zd;E@aE~�(q?iM���.�?�Unknown
�HostMatMul"-gradient_tape/sequential/outputlayer/MatMul_1(1+���D@9+���D@A+���D@I+���D@a�v�p?ik8�,|P�?�Unknown
�HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1��"��H@9��"��H@AZd;�O=D@IZd;�O=D@aCO�;{[p?i
�#3q�?�Unknown
�HostReluGrad".gradient_tape/sequential/hiddenlayer1/ReluGrad(1㥛� �:@9㥛� �:@A㥛� �:@I㥛� �:@a�\_�e?ig<�ކ�?�Unknown
cHostDataset"Iterator::Root(1����M�P@9����M�P@A�O��n�8@I�O��n�8@a��Rr)d?il4Y��?�Unknown
�HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1P��nsA@9P��nsA@A�|?5^z7@I�|?5^z7@a/B>��b?i�rZ���?�Unknown
�HostBiasAddGrad"8gradient_tape/sequential/outputlayer/BiasAdd/BiasAddGrad(1���S��3@9���S��3@A���S��3@I���S��3@a���`?i�i�p��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_3/ResourceApplyGradientDescent(1D�l���1@9D�l���1@AD�l���1@ID�l���1@a^5|��\?i�'�Ȑ��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_4/ResourceApplyGradientDescent(1��Mb81@9��Mb81@A��Mb81@I��Mb81@a�*��[?ie+�U{��?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_5/ResourceApplyGradientDescent(1��|?5�0@9��|?5�0@A��|?5�0@I��|?5�0@a���C[?i;;�C��?�Unknown
ZHostArgMax"ArgMax(1�C�l��,@9�C�l��,@A�C�l��,@I�C�l��,@a�� �lW?i�;ٛ���?�Unknown
gHostStridedSlice"strided_slice(1'1��,@9'1��,@A'1��,@I'1��,@ar��9�W?i�'vob��?�Unknown
�HostReluGrad".gradient_tape/sequential/hiddenlayer2/ReluGrad(1q=
ףp+@9q=
ףp+@Aq=
ףp+@Iq=
ףp+@ay �P=-V?i�~y
�?�Unknown
�HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1?5^�I�(@9?5^�I�(@A?5^�I�(@I?5^�I�(@a��v�
T?i^�Y]~�?�Unknown
�HostResourceApplyGradientDescent"-SGD/SGD/update_1/ResourceApplyGradientDescent(1P��n�(@9P��n�(@AP��n�(@IP��n�(@a­�+T?i5j4��?�Unknown
�HostBiasAddGrad"9gradient_tape/sequential/hiddenlayer1/BiasAdd/BiasAddGrad(1��/ݤ(@9��/ݤ(@A��/ݤ(@I��/ݤ(@a����S?iB�Tu(�?�Unknown
l HostIteratorGetNext"IteratorGetNext(133333s(@933333s(@A33333s(@I33333s(@a��~x��S?i�3T�V2�?�Unknown
�!HostDataset"NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1d;�O��&@9d;�O��&@Ad;�O��&@Id;�O��&@a�l��uR?i�iM��;�?�Unknown
�"HostReadVariableOp".sequential/hiddenlayer1/BiasAdd/ReadVariableOp(1\���(�&@9\���(�&@A\���(�&@I\���(�&@a�Zu��ER?i�$G��D�?�Unknown
�#HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1�G�z.&@9�G�z.&@A�G�z.&@I�G�z.&@ac�<���Q?i	ë�M�?�Unknown
�$HostBiasAddGrad"9gradient_tape/sequential/hiddenlayer2/BiasAdd/BiasAddGrad(1�� �r�%@9�� �r�%@A�� �r�%@I�� �r�%@a�����Q?ii��kV�?�Unknown
w%HostDataset""Iterator::Root::ParallelMapV2::Zip(1�(\��UX@9�(\��UX@Au�V%@Iu�V%@a��t�iQ?i��ټ�^�?�Unknown
e&Host
LogicalAnd"
LogicalAnd(1��(\��$@9��(\��$@A��(\��$@I��(\��$@aE��fr�P?iH&v7g�?�Unknown�
v'HostAssignAddVariableOp"AssignAddVariableOp_2(1�(\���#@9�(\���#@A�(\���#@I�(\���#@az�``�!P?i�V�IHo�?�Unknown
`(HostGatherV2"
GatherV2_1(1�O��n�"@9�O��n�"@A�O��n�"@I�O��n�"@a�����N?i��^��v�?�Unknown
V)HostSum"Sum_2(1㥛� 0"@9㥛� 0"@A㥛� 0"@I㥛� 0"@a�@�fM?i Y�#~�?�Unknown
w*HostReadVariableOp"SGD/Identity/ReadVariableOp(1sh��|�!@9sh��|�!@Ash��|�!@Ish��|�!@a���!rM?i�(�h��?�Unknown
|+HostAssignAddVariableOp"SGD/SGD/AssignAddVariableOp(1���Mb� @9���Mb� @A���Mb� @I���Mb� @a���J?i}�ml��?�Unknown
�,HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1�Zd;_@9�Zd;_@A�Zd;_@I�Zd;_@a'��ZI?i��q��?�Unknown
�-HostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1Zd;�O�@9Zd;�O�@AZd;�O�@IZd;�O�@a�[��H?iF�Tc���?�Unknown
v.HostCast"$sparse_categorical_crossentropy/Cast(1�Zd�@9�Zd�@A�Zd�@I�Zd�@a��=rH?i�J�?���?�Unknown
�/HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1��x�&1@9��x�&1@A��x�&1@I��x�&1@a����E?i�\���?�Unknown
t0HostAssignAddVariableOp"AssignAddVariableOp(1T㥛� @9T㥛� @AT㥛� @IT㥛� @a\=G��E?iSa瘩�?�Unknown
�1HostReadVariableOp"-sequential/hiddenlayer2/MatMul/ReadVariableOp(1�v��/@9�v��/@A�v��/@I�v��/@aT���`)E?i���?��?�Unknown
�2HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_float_Cast(1D�l���@9D�l���@AD�l���@ID�l���@a>p��C?i�(��޳�?�Unknown
y3HostReadVariableOp"SGD/Identity_1/ReadVariableOp(1����K@9����K@A����K@I����K@a^�H��B?i6T��?�Unknown
�4HostReadVariableOp",sequential/outputlayer/MatMul/ReadVariableOp(1y�&1�@9y�&1�@Ay�&1�@Iy�&1�@a$"���RB?i�KV�(��?�Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_4(1��S㥛@9��S㥛@A��S㥛@I��S㥛@a�\�EB?i֓W���?�Unknown
X6HostCast"Cast_2(1�����@9�����@A�����@I�����@a����A?i�Zb���?�Unknown
u7HostReadVariableOp"div_no_nan/ReadVariableOp(1�I+@9�I+@A�I+@I�I+@ab���/@?i)}�P	��?�Unknown
X8HostEqual"Equal(1ˡE��}@9ˡE��}@AˡE��}@IˡE��}@a8�F<?i������?�Unknown
�9HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1��K7��@9��K7��@A��K7��@I��K7��@ax�ԃ;?i��8����?�Unknown
b:HostDivNoNan"div_no_nan_1(1%��C@9%��C@A%��C@I%��C@a_�U���9?iQ/ן2��?�Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_3(1X9��v�@9X9��v�@AX9��v�@IX9��v�@azƎӸ�9?i*��g��?�Unknown
�<HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1-����@9-����@A-����@I-����@a\:�8r�8?iq�8����?�Unknown
�=HostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1)\���XC@9)\���XC@A��|?5^@I��|?5^@a�sL�8?i�NB���?�Unknown
�>HostReadVariableOp"-sequential/hiddenlayer1/MatMul/ReadVariableOp(1+���@9+���@A+���@I+���@a>�e�+8?i��˗��?�Unknown
�?HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1NbX9�@9NbX9�@ANbX9�@INbX9�@aO����27?iѺ�#~��?�Unknown
�@HostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1��ʡE@9��ʡE@A��ʡE@I��ʡE@a�(k�`�6?i6H PY��?�Unknown
VAHostCast"Cast(1���x�&@9���x�&@A���x�&@I���x�&@aW�C���6?i�P�a1��?�Unknown
wBHostReadVariableOp"div_no_nan_1/ReadVariableOp(1�(\���@9�(\���@A�(\���@I�(\���@a�E`Ә6?i2Q|��?�Unknown
`CHostDivNoNan"
div_no_nan(1��Q�@9��Q�@A��Q�@I��Q�@a�D�Z�5?i�����?�Unknown
vDHostAssignAddVariableOp"AssignAddVariableOp_1(1�O��n	@9�O��n	@A�O��n	@I�O��n	@a���OC4?i��FJ��?�Unknown
�EHostReadVariableOp".sequential/hiddenlayer2/BiasAdd/ReadVariableOp(1�O��n@9�O��n@A�O��n@I�O��n@aF���1?iq����?�Unknown
TFHostMul"Mul(1o��ʡ@9o��ʡ@Ao��ʡ@Io��ʡ@a��2��{1?iq�v����?�Unknown
XGHostCast"Cast_3(1�� �rh@9�� �rh@A�� �rh@I�� �rh@ah5��W~0?iXtX���?�Unknown
�HHostReadVariableOp"-sequential/outputlayer/BiasAdd/ReadVariableOp(1�l����@9�l����@A�l����@I�l����@a`�x��.?i���F���?�Unknown
aIHostIdentity"Identity(1��"��~ @9��"��~ @A��"��~ @I��"��~ @a�R*��*?i���X��?�Unknown�
yJHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1w��/��?9w��/��?Aw��/��?Iw��/��?a��/��(?i� ���?�Unknown
XKHostCast"Cast_4(17�A`���?97�A`���?A7�A`���?I7�A`���?a��׸�(?iˌ�{v��?�Unknown
wLHostReadVariableOp"div_no_nan/ReadVariableOp_1(1;�O��n�?9;�O��n�?A;�O��n�?I;�O��n�?awt3F�(?i     �?�Unknown2CPU