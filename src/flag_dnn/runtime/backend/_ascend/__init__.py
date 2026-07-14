from flag_dnn.runtime.backend.backend_utils import VendorInfoBase


vendor_info = VendorInfoBase(
    vendor_name="ascend",
    device_name="npu",
    device_query_cmd="npu-smi info",
    dispatch_key="PrivateUse1",
    triton_extra_name="cann",
)

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
