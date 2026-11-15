"""基础行为与本机沙箱分组验收，均不调用真实模型。"""
import argparse
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="单独运行真实 Windows 沙箱边界与兼容性验收")
    native = parser.parse_args().native
    tests = flatten(unittest.defaultTestLoader.discover(str(ROOT / "tests")))
    selected = [test for test in tests if ("test_sandbox_native." in test.id()) == native]
    if native:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(selected))
    else:
        import test_permissions
        previous = os.environ.get("NAMI_WORKSPACE")
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.environ["NAMI_WORKSPACE"] = directory
                selected.extend(unittest.FunctionTestCase(getattr(test_permissions, name))
                                for name in sorted(dir(test_permissions)) if name.startswith("test_"))
                result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(selected))
        finally:
            if previous is None:
                os.environ.pop("NAMI_WORKSPACE", None)
            else:
                os.environ["NAMI_WORKSPACE"] = previous
    # 显式验收本机边界时，跳过不能作为验收成功。
    return 0 if result.wasSuccessful() and not result.skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
