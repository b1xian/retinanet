from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='retinanet',
    version='0.1',
    description='Fast and accurate single shot object detector',
    author = 'NVIDIA Corporation',
    author_email='fchabert@nvidia.com',
    packages=['retinanet', 'retinanet.backbones'],
    ext_modules=[
        # 扩展包的名字，供python导入import时使用
        # 编译以下4个文件，生成retinanet._C
        CUDAExtension('retinanet._C', ['csrc/extensions.cpp', 'csrc/engine.cpp', 'csrc/cuda/decode.cu', 'csrc/cuda/nms.cu'],
        # 编译扩展包的命令参数(额外编译选项)
        extra_compile_args={
            'cxx': ['-std=c++11', '-O2', '-Wall'],
            'nvcc': [
                '-std=c++11', '--expt-extended-lambda', '--use_fast_math', '-Xcompiler', '-Wall',
                '-gencode=arch=compute_60,code=sm_60', '-gencode=arch=compute_61,code=sm_61',
                '-gencode=arch=compute_70,code=sm_70', '-gencode=arch=compute_72,code=sm_72',
                '-gencode=arch=compute_75,code=sm_75', '-gencode=arch=compute_75,code=compute_75'
            ],
        },
        # libraries 库名（不是文件名称或路径）的组成的列表
        libraries=['nvinfer', 'nvinfer_plugin', 'nvonnxparser'])
    ],
    cmdclass={'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)},
    install_requires=[
        'torch>=1.0.0a0',
        'torchvision',
        'apex @ git+https://github.com/NVIDIA/apex',
        'pycocotools @ git+https://github.com/nvidia/cocoapi.git#subdirectory=PythonAPI',
        'pillow',
        'requests',
    ],
    # console_scripts 指明了命令行工具的名称；在“retinanet=retinanet.main:main”中，等号前面指明了工具包的名称，等号后面的内容指明了程序的入口地址。
    entry_points = {'console_scripts': ['retinanet=retinanet.main:main']}
)
