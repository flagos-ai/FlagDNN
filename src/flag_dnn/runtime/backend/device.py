# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shlex
import subprocess
import threading

import torch  # noqa: F401

from .. import backend, error
from ..common import vendors

UNSUPPORT_FP64 = [
    vendors.CAMBRICON,
    vendors.ILUVATAR,
    vendors.KUNLUNXIN,
    vendors.MTHREADS,
    vendors.AIPU,
    vendors.ASCEND,
    vendors.TSINGMICRO,
    vendors.SUNRISE,
]
UNSUPPORT_BF16 = [
    vendors.AIPU,
    vendors.SUNRISE,
]
UNSUPPORT_INT64 = [
    vendors.AIPU,
    vendors.TSINGMICRO,
    vendors.SUNRISE,
]

DEVICE_QUERY_TIMEOUT_SECONDS = 5.0


# A singleton class to manage device context.
class DeviceDetector(object):
    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DeviceDetector, cls).__new__(cls)
            return cls._instance

    def __init__(self, vendor_name=None):
        cls = type(self)
        with cls._lock:
            if cls._instance is not self:
                raise RuntimeError("stale DeviceDetector instance; retry")
            if getattr(self, "initialized", False):
                return
            try:
                self._initialize(vendor_name)
            except BaseException:
                self.__dict__.clear()
                if cls._instance is self:
                    cls._instance = None
                raise
            self.initialized = True

    def _initialize(self, vendor_name=None):
        # A list of all available vendor names.
        self.vendor_list = vendors.get_all_vendors().keys()

        # A dataclass instance, get the vendor information
        # based on the provided or default vendor name.
        self.info = self.get_vendor(vendor_name)

        # vendor_name is like 'nvidia', device_name is like 'cuda'.
        self.vendor_name = self.info.vendor_name
        self.name = self.info.device_name
        self.vendor = vendors.get_all_vendors()[self.vendor_name]
        self.dispatch_key = (
            self.name.upper()
            if self.info.dispatch_key is None
            else self.info.dispatch_key
        )
        self.device_count = backend.gen_torch_device_object(
            self.vendor_name
        ).device_count()
        self.support_fp64 = self.vendor not in UNSUPPORT_FP64
        self.support_bf16 = self.vendor not in UNSUPPORT_BF16
        self.support_int64 = self.vendor not in UNSUPPORT_INT64

    def get_vendor(self, vendor_name=None) -> tuple:
        if vendor_name is not None:
            if vendor_name not in self.vendor_list:
                raise ValueError(f"Unsupported vendor name: {vendor_name}")
            return backend.get_vendor_info(vendor_name)

        # Try to get the vendor name from a quick special
        # command like 'torch.mlu'.
        vendor_from_env = self._get_vendor_from_env()
        if vendor_from_env is not None:
            return backend.get_vendor_info(vendor_from_env)

        vendor_from_runtime = self._get_vendor_from_quick_cmd()
        if vendor_from_runtime is not None:
            return backend.get_vendor_info(vendor_from_runtime)
        try:
            # Obtaining a vendor_info from the methods provided
            # by torch or triton, but is not currently implemented.
            return self._get_vendor_from_lib()
        except Exception:
            return self._get_vendor_from_sys()

    def _get_vendor_from_quick_cmd(self):
        cmd = {
            "cambricon": "mlu",
            "mthreads": "musa",
            "iluvatar": "corex",
            "ascend": "npu",
            "sunrise": "ptpu",
        }
        registered_flags = set()
        for vendor_name, flag in cmd.items():
            device_api = getattr(torch, flag, None)
            if device_api is None:
                continue
            registered_flags.add(flag)
            if self._device_api_available(device_api):
                return vendor_name
        try:
            import torch_npu

            for vendor_name, flag in cmd.items():
                if flag in registered_flags:
                    continue
                device_api = getattr(torch, flag, None)
                if device_api is not None:
                    if self._device_api_available(device_api):
                        return vendor_name
                    continue
                device_api = getattr(torch_npu, flag, None)
                if device_api is not None and self._device_api_available(
                    device_api
                ):
                    return vendor_name
        except Exception:
            pass
        return None

    @staticmethod
    def _device_api_available(device_api):
        is_available = getattr(device_api, "is_available", None)
        if callable(is_available):
            try:
                return bool(is_available())
            except Exception:
                return False

        device_count = getattr(device_api, "device_count", None)
        if callable(device_count):
            try:
                return int(device_count()) > 0
            except Exception:
                return False
        return True

    def _get_vendor_from_env(self):
        device_from_evn = os.environ.get("DNN_VENDOR")
        return (
            None
            if device_from_evn not in self.vendor_list
            else device_from_evn
        )

    def _get_vendor_from_sys(self):
        vendor_infos = list(backend.get_vendor_infos())
        results = [None] * len(vendor_infos)

        def runcmd(index, single_info):
            device_query_cmd = single_info.device_query_cmd
            try:
                device_api = getattr(torch, single_info.device_name, None)
                if device_api is not None and not self._device_api_available(
                    device_api
                ):
                    return
                cmd_args = shlex.split(device_query_cmd)
                result = subprocess.run(
                    cmd_args,
                    capture_output=True,
                    text=True,
                    timeout=DEVICE_QUERY_TIMEOUT_SECONDS,
                )
                if result.returncode == 0:
                    results[index] = single_info
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass

        threads = []
        for index, single_info in enumerate(vendor_infos):
            # Get the vendor information by running system commands.
            thread = threading.Thread(
                target=runcmd,
                args=(index, single_info),
                daemon=True,
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        for single_info in results:
            if single_info is not None:
                return single_info
        error.device_not_found()

    def get_vendor_name(self):
        return self.vendor_name

    def _get_vendor_from_lib(self):
        # Reserve the associated interface for triton or torch
        # although they are not implemented yet.
        # try:
        #     return triton.get_vendor_info()
        # except Exception:
        #     return torch.get_vendor_info()
        raise RuntimeError("The method is not implemented")
