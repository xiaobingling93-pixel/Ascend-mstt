def _tensor_bytes_view_cpu(t: torch.Tensor, logger=None, tag: str = "crc"):
    """
    返回 t 在当前 dtype 下的原始字节视图（优先零拷贝）。
    需保证：t 已在 CPU 且是 contiguous。
    可能返回 memoryview 或 bytes（兜底拷贝），均可被 zlib.crc32 接受。
    """

    # 直接拿底层 storage（PyTorch 提供的无类型存储，单字节步长）
    # nbytes = t.numel() * t.element_size()
    # storage = t.untyped_storage()
    # mv = memoryview(storage)  # 这就是 u8 视图
    # return mv[:nbytes]        # 截到有效字节长度

    # print("123124")
    def _log(event: str, **kv):
        msg = f"[{tag}] {_safe(event)} | " + " ".join(f"{k}={_safe(v)}" for k, v in kv.items())
        try:
            if logger is not None:
                logger.debug(msg)
            else:
                print(msg)
        except Exception:
            # 避免日志本身抛错影响主流程
            try:
                print(msg)
            except Exception:
                pass

    def _safe(x):
        try:
            return str(x)
        except Exception:
            return f"<unrepr {type(x).__name__}>"

    nbytes = t.numel() * t.element_size()
    byte_offset = t.storage_offset() * t.element_size()

    # _log(
    #     "enter",
    #     dtype=t.dtype,
    #     device=t.device,
    #     shape=tuple(t.shape),
    #     contiguous=t.is_contiguous(),
    #     numel=t.numel(),
    #     elem_size=t.element_size(),
    #     nbytes=nbytes,
    #     storage_offset_bytes=byte_offset,
    # )

    if nbytes == 0:
        # _log("empty_tensor", action="return_empty_memoryview")
        return memoryview(b"")

    # A) 直接对 UntypedStorage 建立 memoryview
    storage = t.untyped_storage()
    # try:
    #     mv = memoryview(storage)  # 有些发行版支持
    #     mv_sliced = mv[byte_offset: byte_offset + nbytes]
    #     # _log(
    #     #     "path_A_success",
    #     #     mv_type=type(mv_sliced).__name__,
    #     #     mv_len=len(mv_sliced),
    #     #     head16=bytes(mv_sliced[:16]).hex() if len(mv_sliced) else "",
    #     # )
    #     return mv_sliced
    # except Exception as e1:
    #     pass
    #     # _log("path_A_failed", err=repr(e1))

    # # B) 视作 uint8 再取 storage（依然零拷贝）
    # try:
    #     t_u8 = t.view(torch.uint8)
    #     st_u8 = t_u8.untyped_storage()
    #     mv2 = memoryview(st_u8)  # uint8 下 offset 单位就是字节
    #     off_elems = byte_offset
    #     mv2_sliced = mv2[off_elems: off_elems + nbytes]
    #     # _log(
    #     #     "path_B_success",
    #     #     mv_type=type(mv2_sliced).__name__,
    #     #     mv_len=len(mv2_sliced),
    #     #     head16=bytes(mv2_sliced[:16]).hex() if len(mv2_sliced) else "",
    #     # )
    #     return mv2_sliced
    # except Exception as e2:
    #     # _log("path_B_failed", err=repr(e2))
    #     pass

    # C) ctypes 指针构造 memoryview（零拷贝 FFI）
    import ctypes
    try:
        addr = storage.data_ptr() + byte_offset
        buf = (ctypes.c_ubyte * nbytes).from_address(addr)
        mv3 = memoryview(buf)
        # _log(
        #     "path_C_success",
        #     mv_type=type(mv3).__name__,
        #     mv_len=len(mv3),
        #     head16=bytes(mv3[:16]).hex() if len(mv3) else "",
        #     addr_hex=hex(addr),
        # )
        return mv3
    except Exception as e3:
        _log("path_C_failed", err=repr(e3))
        pass
    # D) 兜底拷贝一份 bytes，确保不崩
    try:
        data = ctypes.string_at(storage.data_ptr() + byte_offset, nbytes)
        _log(
            "path_D_copy_success",
            bytes_len=len(data),
            head16=data[:16].hex() if len(data) else "",
        )
        return data  # bytes 也可直接用于 zlib.crc32
    except Exception as e4:
        _log("path_D_copy_failed", err=repr(e4))
        raise RuntimeError(
            f"failed to obtain tensor bytes view; "
            f" | C:{e3!r} | D:{e4!r}"
        )

    # E) 兜底拷贝一份 bytes，确保不崩
    try:
        if t.dtype == torch.bfloat16:
            t = t.float()
        data = t.numpy()
        # data = ctypes.string_at(storage.data_ptr() + byte_offset, nbytes)
        # _log(
        #     "path_D_copy_success",
        #     bytes_len=len(data)
        #     "",
        # )
        return data  # bytes 也可直接用于 zlib.crc32
    except Exception as e5:
        _log("path_E_copy_failed", err=repr(e5))
        raise RuntimeError(
            f"failed to obtain tensor bytes view; "
            f" | C:{e3!r} | D:{e4!r} | D:{e5!r}"
        )
