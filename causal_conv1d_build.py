# causal_conv1d_build.py
import os
import re
import sys
import shutil
import tempfile
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from packaging.version import parse, Version

# Re-export setuptools' backend but override a few hooks
try:
    from setuptools import build_meta as _orig  # type: ignore
    from setuptools.build_meta import *  # noqa: F401,F403
except Exception as e:
    raise RuntimeError(f"Failed to import setuptools.build_meta: {e}")

PACKAGE_NAME = "causal_conv1d"
BASE_WHEEL_URL = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/{tag_name}/{wheel_name}"
)

# Env toggles
FORCE_BUILD = os.getenv("CAUSAL_CONV1D_FORCE_BUILD", "FALSE").upper() == "TRUE"

# ---- Utilities -----------------------------------------------------------------

def _safe_import_torch():
    try:
        import torch  # noqa
        return torch
    except Exception:
        return None

def _get_platform_tag():
    import platform
    m = platform.machine().lower()
    if sys.platform.startswith("linux"):
        if m in ("x86_64", "amd64"):
            return "linux_x86_64"
        if m in ("aarch64", "arm64"):
            return "linux_aarch64"
    elif sys.platform == "darwin":
        # Upstream only publishes CPU wheels for mac; we still use a tag for completeness.
        if m in ("arm64", "aarch64"):
            return "macosx_11_0_arm64"
        return "macosx_10_15_x86_64"
    elif sys.platform == "win32":
        return "win_amd64"
    raise RuntimeError(f"Unsupported platform: {sys.platform} {m}")

def _read_version_from_package():
    here = Path(__file__).resolve().parent
    init_py = here / "causal_conv1d" / "__init__.py"
    text = init_py.read_text(encoding="utf-8")
    m = re.search(r"^__version__\s*=\s*(.*)$", text, re.MULTILINE)
    ver = eval(m.group(1))  # literal eval but shorter; file is trusted in build
    local = os.environ.get("CAUSAL_CONV1D_LOCAL_VERSION")
    return f"{ver}+{local}" if local else str(ver)

def _compute_upstream_filename(torch, version_str):
    """
    Wheel filename scheme:
    causal_conv1d-{ver}+{cu|hip}{X}torch{MAJOR.MINOR}cxx11abi{TRUE|FALSE}-{cpXY}-{cpXY}-{plat}.whl
    """
    pyver = f"cp{sys.version_info.major}{sys.version_info.minor}"
    plat = _get_platform_tag()

    torch_version_raw = parse(torch.__version__)
    torch_mm = f"{torch_version_raw.major}.{torch_version_raw.minor}"

    if getattr(torch, "version", None) and torch.version.hip:
        hip_v = torch.version.hip.split()[-1].replace("-", "+")
        hip = parse(hip_v)
        gpu_compute = f"{hip.major}{hip.minor}"
        cuda_or_hip = "hip"
    else:
        tv_raw = parse(torch.version.cuda) if (getattr(torch, "version", None) and torch.version.cuda) else None
        if not tv_raw:
            return None  # CPU-only torch -> no matching upstream GPU wheel
        # Normalize to 11.8 or 12.3 family
        tv = parse("11.8") if tv_raw.major == 11 else parse("12.3")
        gpu_compute = f"{tv.major}"
        cuda_or_hip = "cu"

    try:
        cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
    except Exception:
        cxx11_abi = "FALSE"

    return (
        f"{PACKAGE_NAME}-{version_str}+{cuda_or_hip}{gpu_compute}"
        f"torch{torch_mm}cxx11abi{cxx11_abi}-{pyver}-{pyver}-{plat}.whl"
    )

def _download_upstream_wheel(package_version, out_dir):
    """
    Try to download the upstream wheel matching the visible torch.
    Return the basename if successful, else None.
    """
    torch = _safe_import_torch()
    if torch is None:
        return None

    wheel_name = _compute_upstream_filename(torch, package_version)
    if not wheel_name:
        return None

    url = os.getenv(
        "CAUSAL_CONV1D_WHEEL_URL",
        BASE_WHEEL_URL.format(tag_name=f"v{package_version}", wheel_name=wheel_name),
    )

    tmpdir = tempfile.mkdtemp()
    tmp_path = Path(tmpdir) / wheel_name
    try:
        print(f"[causal_conv1d_build] Trying upstream wheel: {url}")
        urllib.request.urlretrieve(url, tmp_path)
        dest = Path(out_dir) / wheel_name
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
        shutil.rmtree(tmpdir, ignore_errors=True)

def _read_metadata_version(metadata_directory):
    if not metadata_directory:
        return None
    metas = list(Path(metadata_directory).glob("*.dist-info/METADATA"))
    if not metas:
        return None
    text = metas[0].read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^Version:\s*([^\s]+)", text, re.MULTILINE)
    return m.group(1) if m else None

def _wheel_version_from_filename(wheel_filename: str) -> str:
    # causal_conv1d-1.5.2+cu12...-cp310-cp310-linux_x86_64.whl
    base = Path(wheel_filename).name
    # project - version - (tags...).whl
    parts = base.split("-")
    if len(parts) < 3:
        return ""
    return parts[1]

# ---- Build-time requirements ----------------------------------------------------

def get_requires_for_build_wheel(config_settings=None):
    """
    Ensure Torch is installed in the isolated build env *before* metadata/build.
    uv calls this prior to prepare_metadata_for_build_wheel.
    """
    req = ["torch"]
    try:
        extra = _orig.get_requires_for_build_wheel(config_settings) or []
    except Exception:
        extra = []
    # dedupe while preserving order
    seen = set()
    out = []
    for x in req + extra:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def get_requires_for_build_editable(config_settings=None):
    return get_requires_for_build_wheel(config_settings)

def get_requires_for_build_sdist(config_settings=None):
    # Optional: if sdist metadata also needs torch to compute a local version
    return ["torch"]

# ---- PEP 517 hooks we override --------------------------------------------------

def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """
    Emit metadata whose Version matches the wheel we intend to return.
    If we can download the upstream wheel now, extract its .dist-info.
    Otherwise, mark that we couldn't, and fall back to setuptools' metadata.
    """
    if FORCE_BUILD:
        return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)

    pkg_version = _read_version_from_package()

    # Try to get an upstream wheel NOW; if torch is present this should work.
    tmpdir = tempfile.mkdtemp()
    try:
        wheel_name = _download_upstream_wheel(pkg_version, tmpdir)
        if wheel_name:
            wheel_path = Path(tmpdir) / wheel_name
            with zipfile.ZipFile(wheel_path) as zf:
                # Find the dist-info dir
                distinfos = sorted(
                    {n.split("/")[0] for n in zf.namelist() if n.endswith(".dist-info/METADATA")}
                )
                if not distinfos:
                    # Unexpected; fall back
                    return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)
                distinfo_root = distinfos[0]
                # Extract only that folder
                members = [m for m in zf.namelist() if m.startswith(distinfo_root)]
                zf.extractall(metadata_directory, members)
                # Stash the expected version (helps build_wheel validate)
                try:
                    meta_text = (Path(metadata_directory) / distinfo_root / "METADATA").read_text(encoding="utf-8", errors="ignore")
                    m = re.search(r"^Version:\s*([^\s]+)", meta_text, re.MULTILINE)
                    if m:
                        (Path(metadata_directory) / ".expected_version").write_text(m.group(1), encoding="utf-8")
                except Exception:
                    pass
                # Record that metadata used upstream wheel
                (Path(metadata_directory) / ".used_upstream_wheel").write_text("1", encoding="utf-8")
                return distinfo_root

        # Couldn’t fetch upstream wheel at metadata time: remember this decision
        (Path(metadata_directory) / ".no_upstream_wheel").write_text("1", encoding="utf-8")
        return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """
    Return the upstream wheel iff metadata used it (and versions match).
    Otherwise build from source so filename version matches metadata.
    """
    if FORCE_BUILD:
        print("[causal_conv1d_build] FORCE_BUILD=TRUE -> building from source.")
        return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)

    marker_no_upstream = Path(metadata_directory or ".", ".no_upstream_wheel")
    marker_used_upstream = Path(metadata_directory or ".", ".used_upstream_wheel")
    expected_ver = None

    # Try read expected version written during metadata (if any)
    try:
        expected_path = Path(metadata_directory or ".") / ".expected_version"
        if expected_path.exists():
            expected_ver = expected_path.read_text(encoding="utf-8").strip()
    except Exception:
        expected_ver = None

    # If metadata couldn't use upstream wheel, DON'T return a +local wheel now.
    if marker_no_upstream.exists() and not marker_used_upstream.exists():
        print("[causal_conv1d_build] Metadata didn’t use upstream wheel; building from source for version consistency.")
        return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)

    # If metadata *did* use upstream wheel, try to return the same flavor.
    pkg_version = _read_version_from_package()
    wheel_name = _download_upstream_wheel(pkg_version, wheel_directory)
    if wheel_name:
        if expected_ver:
            fn_ver = _wheel_version_from_filename(wheel_name)
            if fn_ver != expected_ver:
                # Don’t leave a mismatched wheel behind; build from source instead.
                try:
                    Path(wheel_directory, wheel_name).unlink()
                except Exception:
                    pass
                print(f"[causal_conv1d_build] Wheel version {fn_ver} != metadata {expected_ver}; building from source.")
                return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)
        else:
            # As a fallback, compare with live metadata (if present)
            meta_ver_now = _read_metadata_version(metadata_directory)
            if meta_ver_now:
                fn_ver = _wheel_version_from_filename(wheel_name)
                if fn_ver != meta_ver_now:
                    try:
                        Path(wheel_directory, wheel_name).unlink()
                    except Exception:
                        pass
                    print(f"[causal_conv1d_build] Wheel version {fn_ver} != metadata {meta_ver_now}; building from source.")
                    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)
        return wheel_name

    # Couldn’t fetch upstream wheel at build time → build from source
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)

def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    # Editable installs: delegate to setuptools (no upstream wheel juggling).
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)
