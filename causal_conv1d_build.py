# causal_conv1d_build.py
import os
import shutil
import sys
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
import zipfile
from packaging.version import parse, Version

# Delegate all other hooks to setuptools' backend
try:
    from setuptools.build_meta import *  # noqa: F401,F403
    import setuptools.build_meta as _orig
except Exception as e:
    raise RuntimeError(f"Failed to import setuptools.build_meta: {e}")

PACKAGE_NAME = "causal_conv1d"
BASE_WHEEL_URL = "https://github.com/Dao-AILab/causal-conv1d/releases/download/{tag_name}/{wheel_name}"

# Env toggles (same semantics as your setup.py)
FORCE_BUILD = os.getenv("CAUSAL_CONV1D_FORCE_BUILD", "FALSE") == "TRUE"

def _safe_import_torch():
    try:
        import torch  # noqa
        return torch
    except Exception:
        return None

# in causal_conv1d_build.py
def _get_platform_tag():
    import platform
    m = platform.machine().lower()
    if sys.platform.startswith("linux"):
        # Upstream uses 'linux_x86_64' and 'linux_aarch64'
        if m in ("x86_64", "amd64"):
            return "linux_x86_64"
    raise RuntimeError(f"Unsupported platform: {sys.platform} {m}")

def _compute_upstream_filename(torch, version_str):
    """
    Build upstream wheel filename using your scheme:
    {name}-{ver}+{cu|hip}{X}torch{MAJOR.MINOR}cxx11abi{TRUE|FALSE}-{py}-{py}-{plat}.whl
    """
    pyver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    plat = _get_platform_tag()

    # Torch major.minor
    torch_version_raw = parse(torch.__version__)
    torch_mm = f"{torch_version_raw.major}.{torch_version_raw.minor}"

    # HIP or CUDA family from the torch build
    if torch.version.hip:
        # Use HIP version from torch build, e.g. "6.0"
        hip_v = torch.version.hip.split()[-1].replace("-", "+")
        hip = parse(hip_v)
        gpu_compute = f"{hip.major}{hip.minor}"
        cuda_or_hip = "hip"
    else:
        # Use CUDA major family; normalize to 11.8 or 12.3 family (like your code)
        tv_raw = parse(torch.version.cuda) if torch.version.cuda else None
        if not tv_raw:
            return None  # CPU-only torch: no upstream GPU wheel to download
        tv = parse("11.8") if tv_raw.major == 11 else parse("12.3")
        gpu_compute = f"{tv.major}"
        cuda_or_hip = "cu"

    # Respect cxx11 abi flag if torch exposes it; default FALSE
    try:
        cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    except Exception:
        cxx11_abi = "FALSE"

    wheel_filename = (
        f"{PACKAGE_NAME}-{version_str}+{cuda_or_hip}{gpu_compute}"
        f"torch{torch_mm}cxx11abi{cxx11_abi}-{pyver}-{pyver}-{plat}.whl"
    )
    return wheel_filename

def _download_upstream_wheel(package_version, wheel_directory):
    """
    Try downloading the upstream wheel matching local torch/python/platform.
    Returns basename if successful, else None.
    """
    torch = _safe_import_torch()
    if torch is None:
        return None  # no torch in build env (e.g., metadata-only or CPU) -> cannot match a GPU wheel

    wheel_name = _compute_upstream_filename(torch, package_version)
    if not wheel_name:
        return None

    url = BASE_WHEEL_URL.format(tag_name=f"v{package_version}", wheel_name=wheel_name)

    # allow override via env for testing
    url = os.getenv("CAUSAL_CONV1D_WHEEL_URL", url)

    tmpdir = tempfile.mkdtemp()
    tmp_path = Path(tmpdir) / wheel_name
    try:
        print(f"[causal_conv1d_build] Trying upstream wheel: {url}")
        urllib.request.urlretrieve(url, tmp_path)
        dest = Path(wheel_directory) / wheel_name
        shutil.move(str(tmp_path), str(dest))
        print(f"[causal_conv1d_build] Downloaded: {dest.name}")
        return dest.name
    except urllib.error.HTTPError as e:
        print(f"[causal_conv1d_build] Upstream wheel not found ({e.code}). Falling back.")
        return None
    except Exception as e:
        print(f"[causal_conv1d_build] Failed to fetch upstream wheel: {e}. Falling back.")
        return None
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

def _read_version_from_package():
    # Mirrors setup.py get_package_version() but avoids importing the package.
    here = Path(__file__).resolve().parent
    init_py = here / "causal_conv1d" / "__init__.py"
    text = init_py.read_text(encoding="utf-8")
    import re, ast
    m = re.search(r"^__version__\s*=\s*(.*)$", text, re.MULTILINE)
    ver = ast.literal_eval(m.group(1))
    local = os.environ.get("CAUSAL_CONV1D_LOCAL_VERSION")
    return f"{ver}+{local}" if local else str(ver)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """
    Ensure the metadata version matches the wheel we'll return in build_wheel().
    We try to download the upstream wheel first; if successful, we extract its
    .dist-info directory to metadata_directory and return that folder name.
    Otherwise, fall back to setuptools' default behavior.
    """
    if FORCE_BUILD:
        # We'll build from source; setuptools will keep version consistent.
        return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)

    pkg_version = _read_version_from_package()
    tmpdir = tempfile.mkdtemp()
    try:
        wheel_name = _download_upstream_wheel(pkg_version, tmpdir)
        if wheel_name:
            wheel_path = Path(tmpdir) / wheel_name
            with zipfile.ZipFile(wheel_path) as zf:
                # Find the *.dist-info root inside the wheel
                distinfos = sorted(
                    {n.split("/")[0] for n in zf.namelist() if n.endswith(".dist-info/METADATA")}
                )
                if not distinfos:
                    # Unexpected; fall back
                    return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)
                distinfo_root = distinfos[0]
                # Extract only that folder to the metadata dir
                members = [m for m in zf.namelist() if m.startswith(distinfo_root)]
                zf.extractall(metadata_directory, members)
                # Return the dist-info directory name (per PEP 517)
                return distinfo_root
        # If we couldn’t get a wheel, fall back
        return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


# ---- PEP 517 hook override ----
def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """
    Try to return the official upstream wheel. If unavailable, delegate to setuptools.
    """
    if FORCE_BUILD:
        print("[causal_conv1d_build] FORCE_BUILD=TRUE -> building from source.")
        return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)

    pkg_version = _read_version_from_package()
    got = _download_upstream_wheel(pkg_version, wheel_directory)
    if got:
        # Success: return the filename we put into wheel_directory
        return got

    # No upstream wheel matched; build from source (CUDA/ROCm toolchain required)
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)

# Everything else (sdist, metadata, etc.) is inherited from setuptools.build_meta
