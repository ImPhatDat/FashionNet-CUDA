# CSC14120 - Parallel Programming - 21KHMT1

## Final Project - Optimizing Artificial Neural Networks (ANN)

## Group information

Group name: HK(D)T

| Student ID | Name |
| ----------|-------|
| 21127050 | Trần Nguyên Huân |
| 21127181 | Nguyễn Nhật Tiến |
| 21127240 | Nguyễn Phát Đạt |

## Set up
- CUDA toolkit and nvcc compiler with version `12.2` or higher.
- Download Fashion-MNIST dataset: https://github.com/zalandoresearch/fashion-mnist.
- For stable result, we recommend using Linux distribution (Ubuntu) over Windows (some built-in libraries may be missing on Windows).

## How to use

Compile: Run following command
```bash
nvcc main.cu [any other required *.cu files] -o [output_file] -std=[c++ version] # compile
./[output_file] [command line arguments] # to run
```
Or using our predefined compiled command:

```bash
bash compile.sh # compile sequential training source
bash compile_infer.sh # compile infer source
bash compile_parallel.sh # compile parallel training source
bash compile_parallel_infer.sh # compile parallel infer source
```

Training commandline arguments:

- -d: dỉrectory to dataset
- -e: number of epochs
- -b: batch size
- -l: learning rate
- -p: directory to save checkpoint

Infer commandline arguments:
-p: directory to trained checkpoint.
-i: directory to a raw 28x28 image (input).

### Youtube video link

[Link]()
