# 环境问题记录

## 1. `pip install "lerobot[libero]"` 失败：egl_probe / hf-egl-probe 编译不过

环境：conda env `lerobot`（Python 3.12），lerobot 0.6.1，Ubuntu 22.04 容器。

### 现象

```
Building wheel for egl_probe (pyproject.toml) ... error
  ...
  File "/opt/conda/envs/lerobot/bin/cmake", line 3, in <module>
    from cmake import cmake
  ModuleNotFoundError: No module named 'cmake'
  ...
  subprocess.CalledProcessError: Command '['cmake', '--version']' returned non-zero exit status 1.
ERROR: Failed building wheel for egl_probe
ERROR: Failed building wheel for hf-egl-probe
```

日志里大量的 `Package 'egl_probe.glad' is absent from the packages configuration` 只是 setuptools 警告，与失败无关。

### 原因

- `lerobot[libero]` → `hf-libero` → `egl_probe` / `hf-egl-probe`，这两个只有源码包，安装时要现场编译，`setup.py` 里直接调用 PATH 上的 `cmake`。
- 环境里有两个 cmake：
  - `/opt/conda/envs/lerobot/bin/cmake`：PyPI 的 `cmake` 包（4.x）装的**入口脚本**，内容是 `from cmake import cmake`
  - `/usr/bin/cmake`：apt 装的系统 cmake（3.22）
- conda env 的 bin 在 PATH 最前面，所以先命中入口脚本。
- pip 默认**隔离构建**（build isolation）：编译时在临时环境里只装 setuptools/wheel，看不到 conda env 的 site-packages。入口脚本在这个临时环境里执行 `import cmake` 失败 → `cmake --version` 返回非 0 → 编译失败。
- 在 conda env 里直接 `cmake --version` 是正常的，所以容易误以为"cmake 没问题"。可以这样复现：
  ```bash
  python -I -S -c "import cmake"   # 模拟看不到 site-packages → ModuleNotFoundError
  ```

### 解决

安装期间让系统 cmake 排在 PATH 最前：

```bash
mkdir -p /tmp/syscmake && ln -sf /usr/bin/cmake /tmp/syscmake/cmake
PATH=/tmp/syscmake:$PATH pip install "lerobot[libero]==0.6.1"
rm -rf /tmp/syscmake
pip check
```

- 锁住 `==0.6.1`，避免顺带升级 lerobot。装完核对 torch / numpy / transformers 版本没被改动。
- 不推荐的替代方案：
  - `--no-build-isolation`：会用到 cmake 4.x，它移除了对 `cmake_minimum_required` 很老的 CMakeLists 的兼容，容易出另一个错。
  - 卸载 pip 的 `cmake` 包：它是 lerobot 的依赖（`pip show cmake` → `Required-by: lerobot`）。

## 2. MuJoCo 无头渲染：EGL 不可用，改用 OSMesa

### 现象

```
MUJOCO_GL=egl    → AttributeError: 'NoneType' object has no attribute 'eglQueryString'
MUJOCO_GL=osmesa → AttributeError: 'NoneType' object has no attribute 'glGetError'
```

### 原因

- 容器里没有 libEGL / libOSMesa 系统库。
- EGL 需要 NVIDIA 驱动提供的 `libEGL_nvidia.so`，它由 NVIDIA Container Toolkit 注入，前提是 `NVIDIA_DRIVER_CAPABILITIES` 包含 `graphics`。容器设置的是 `compute,utility`，所以只装 apt 的 `libegl1` 也不行（报 `EGLError`）。

### 解决

```bash
apt-get install -y --no-install-recommends libosmesa6 libegl1 libglvnd0 libgles2
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
```

OSMesa 是 CPU 软件渲染，能出图但比 GPU 渲染慢。想用 EGL，需要在启动容器时加上 `graphics` 能力（`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`），这得由创建容器的一方来改。

## 3. 第一次 `import libero` 卡在交互提问

```
Do you want to specify a custom path for the dataset folder? (Y/N):
EOFError: EOF when reading a line   # 非交互环境下
```

只有 `~/.libero/config.yaml` 不存在时才会问，生成一次就好：

```bash
echo n | python -c "import libero.libero"
```

另外，场景资产（约 600 个文件）第一次用时会从 HF Hub 下载到 `~/.cache/libero/assets`，需要联网（走代理）。
