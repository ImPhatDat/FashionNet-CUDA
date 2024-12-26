nvcc infer.cu utils/fashion_mnist.cu \
    layer/layer.cu layer/softmax.cu layer/relu.cu layer/dense.cu \
    loss/loss.cu loss/categorical_crossentropy.cu \
    Model/Model.cu \
    metrics/accuracy.cu \
    -o infer_parallel_fp16.out -arch=sm_60