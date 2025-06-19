nvidia-smi # system management interface

# nvcc = nvidia cuda compiler
nvcc -o out hello.cu -run # compiles hello.cu into out
# run executes the compiled out (convenience)

# nsys = nsight systems
!nsys --version
!nsys profile --stats=true ./out
# essentially a cuda program profiler
