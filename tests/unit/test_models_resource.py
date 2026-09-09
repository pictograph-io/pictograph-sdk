class TestSafetensorsHeaderValidation:
    """The dependency-free safetensors header check the download runs."""

    def _valid_file(self, tmp_path):
        import json
        import struct

        header = json.dumps(
            {
                "linear.weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
            }
        ).encode("utf-8")
        payload = b"\x00" * 16
        p = tmp_path / "ok.safetensors"
        p.write_bytes(struct.pack("<Q", len(header)) + header + payload)
        return p

    def test_valid_header_passes(self, tmp_path):
        from pictograph.resources.models import _validate_safetensors_header

        _validate_safetensors_header(self._valid_file(tmp_path))  # no raise

    def test_metadata_only_header_fails(self, tmp_path):
        import json
        import struct

        import pytest

        from pictograph.resources.models import _validate_safetensors_header

        header = json.dumps({"__metadata__": {"pipeline": "x"}}).encode("utf-8")
        p = tmp_path / "empty.safetensors"
        p.write_bytes(struct.pack("<Q", len(header)) + header)
        with pytest.raises(ValueError):
            _validate_safetensors_header(p)

    def test_garbage_file_fails(self, tmp_path):
        import pytest

        from pictograph.resources.models import _validate_safetensors_header

        p = tmp_path / "junk.safetensors"
        p.write_bytes(b"this is not a safetensors file at all")
        with pytest.raises(Exception):
            _validate_safetensors_header(p)
