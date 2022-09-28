# 设置是否开启MKL、GPU、TensorRT，如果要使用TensorRT，必须打开GPU
WITH_GPU=ON

# 按照运行环境设置预测库路径、CUDA库路径、CUDNN库路径、模型路径
BOOST_DIR=/usr/lib/x86_64-linux-gnu/
DP_LIB_DIR=$DEEP_MD_PATH
LIB_DIR=$PWD/paddle_inference/
CUDA_LIB_DIR=/usr/local/cuda-11.0/targets/x86_64-linux/lib/
CUDNN_LIB_DIR=/usr/lib/x86_64-linux-gnu/
MODEL_DIR=$PWD/model.ckpt
OP_DIR=/home/deepmd-kit/source/op/paddle_ops/srcs

# GPU Version
CUSTOM_OPERATOR_FILES="${OP_DIR}/pd_prod_env_mat_multi_devices_cpu.cc;${OP_DIR}/pd_prod_env_mat_multi_devices_cuda.cc;${OP_DIR}/pd_prod_force_se_a_multi_devices_cpu.cc;${OP_DIR}/pd_prod_force_se_a_multi_devices_cuda.cc;${OP_DIR}/pd_prod_virial_se_a_multi_devices_cpu.cc;${OP_DIR}/pd_prod_virial_se_a_multi_devices_cuda.cc;"

# Uncomment this if you wanna use CPU BACKEND
#CUSTOM_OPERATOR_FILES="${OP_DIR}/pd_prod_env_mat_multi_devices_cpu.cc;${OP_DIR}/pd_prod_force_se_a_multi_devices_cpu.cc;${OP_DIR}/pd_prod_virial_se_a_multi_devices_cpu.cc;"

sh run_impl.sh ${LIB_DIR} infer_test ${MODEL_DIR} ${WITH_GPU} ${CUDNN_LIB_DIR} ${CUDA_LIB_DIR} ${CUSTOM_OPERATOR_FILES} ${DP_LIB_DIR} ${BOOST_DIR}
