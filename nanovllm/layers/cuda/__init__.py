import os
import subprocess
import torch
from torch.utils.cpp_extension import load
import shutil

print("!!!! LOADING CUDA !!!!")
# ===================== 1. 自动查询 g++ 版本 =====================
def get_gcc_version():
    try:
        # 使用 -dumpversion 获取简洁的版本号
        result = subprocess.check_output(["g++", "-dumpfullversion", "-dumpversion"], text=True)
        return result.strip()
    except Exception as e:
        return f"Unknown (Error: {e})"

gcc_version = get_gcc_version()
print(f"[CUDA Load] Detected g++ version: {gcc_version}")

# ===================== 2. 设置路径与环境 =====================
CUDA_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(CUDA_DIR, "build")

# 关键修复：手动创建 build 目录，防止 lock 文件写入失败
if not os.path.exists(BUILD_DIR):
    os.makedirs(BUILD_DIR, exist_ok=True)
    print(f"[CUDA Load] Created build directory: {BUILD_DIR}")

# 如果之前有残留的 lock 文件导致死锁，可以根据需要手动清理（可选）
lock_file = os.path.join(BUILD_DIR, "lock")
if os.path.exists(lock_file):
    try:
        os.remove(lock_file)
    except:
        pass

# ===================== 3. 收集源文件 =====================
cuda_sources = []
for root, dirs, files in os.walk(CUDA_DIR):
    if "build" in dirs:
        dirs.remove("build")
    for f in files:
        if f.endswith(".cu") or f.endswith(".cpp"):
            cuda_sources.append(os.path.join(root, f))

if not cuda_sources:
    raise RuntimeError(f"No .cu or .cpp files found in {CUDA_DIR}")

# ===================== 4. 编译参数 =====================
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    computed_arch = f"sm_{major}{minor}"
else:
    computed_arch = "sm_80" # 默认备份

EXTRA_CUDA_FLAGS = [
    "-O3",
    f"-arch={computed_arch}",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math"
]

EXTRA_CFLAGS = [
    "-std=c++17",
    f"-DGCC_VERSION={gcc_version.replace('.', '')}"
]

# ===================== 5. 执行编译 =====================
print(f"[CUDA Load] Compiling {len(cuda_sources)} files for {computed_arch}...")

try:
    project_cuda_ops = load(
        name="project_cuda_ops",
        sources=cuda_sources,
        extra_cuda_cflags=EXTRA_CUDA_FLAGS,
        extra_cflags=EXTRA_CFLAGS,
        build_directory=BUILD_DIR,
        with_cuda=True,
        verbose=True  # 开启详细日志以便排查
    )
    
    # 将编译出的函数注入到当前命名空间，这样可以直接通过 folder.func() 调用
    if project_cuda_ops:
        globals().update({k: v for k, v in project_cuda_ops.__dict__.items() if not k.startswith("__")})
        print("[CUDA Load] Compilation and loading successful.")

except Exception as e:
    print("\n" + "="*50)
    print(f"[CUDA Load] ERROR: Compilation failed!")
    print(f"Message: {e}")
    print("="*50 + "\n")
    # 抛出异常防止程序在算子缺失的情况下继续运行
    raise e
__all__ = ["project_cuda_ops"]