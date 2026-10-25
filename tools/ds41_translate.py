#!/usr/bin/env python3
"""Translate a DeepSeek-V4.1-Flash GGUF set made by the Strata/Maya packers (e.g. the IST-DAS GSQ-RCO quants) into
the layout llama.cpp's deepseek41 arch reads (#31086). Tensor bytes are copied, never requantised.

  - metadata: the packer's keys are renamed (kv_source_layer_ids -> attention.kv_source_layers, ...), the Engram
    layout becomes the PLE keys of conversion/deepseek41.py (absolute head offsets, bucket sizes, multipliers,
    token map) and is checked against the layout args the same way the converter computes it
  - tensors: blk.N.engram_wkv -> engram_kv, engram_q * engram_k -> engram_gate (F32, exact), the engram_* helper
    tensors are dropped, and the per-layer Engram tables (a separate GGUF) are concatenated in layer order into
    per_layer_token_embd, written as the last split so it stays a file of its own

usage: ds41_translate.py <first shard> <engram gguf> <out dir> [--meta <gguf with #31086 keys, to cross-check>]
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np

if 'NO_LOCAL_GGUF' not in __import__('os').environ:
    sys.path.insert(1, str(Path(__file__).parent.parent / 'gguf-py'))
    sys.path.insert(1, str(Path(__file__).parent.parent))
import gguf  # noqa: E402

logger = logging.getLogger("ds41-translate")

# packer keys replaced by #31086's (deepseek41.<key>)
DROP_KEYS = {
    "kv_source_layer_ids", "index_source_layer_ids", "dspark_target_layer_ids",
    "attention.candidate_source_layer_id", "attention.candidate_topk_blocks", "attention.candidate_block_size",
    "engram.present", "engram.layer_ids", "engram.num_embeddings", "engram.file",
    "engram_head_dim", "engram_n_heads", "engram_max_ngram_size", "engram_vocab_size",
    "engram_compressed_vocab_size", "engram_pad_token_id",
}
HELPER_TENSORS = {"engram_token_map.weight", "engram_primes.weight", "engram_offsets.weight", "engram_multipliers.weight"}


class ConcatRows:
    """Rows of several same-typed tensors written back to back as one tensor, streamed from the source mmaps."""
    CHUNK = 1 << 28

    def __init__(self, parts: list[np.ndarray]):
        width = parts[0].shape[-1]
        assert all(p.dtype == np.uint8 and p.ndim == 2 and p.shape[1] == width for p in parts)
        self.parts = parts
        self.shape = (sum(p.shape[0] for p in parts), width)
        self.dtype = np.dtype(np.uint8)
        self.nbytes = self.shape[0] * width

    def tofile(self, fout) -> None:
        for p in self.parts:
            step = max(1, self.CHUNK // p.shape[1])
            for r in range(0, p.shape[0], step):
                np.ascontiguousarray(p[r:r + step]).tofile(fout)

    def byteswap(self, inplace: bool = False):
        raise NotImplementedError


def raw_bytes(t: gguf.ReaderTensor) -> np.ndarray:
    """The tensor's bytes as uint8 rows (numpy order shape, last dim in bytes), still backed by the mmap."""
    shape = [int(n) for n in reversed(t.shape)]
    rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    flat = t.data.reshape(-1).view(np.uint8)
    assert flat.nbytes == t.n_bytes, t.name
    return flat.reshape(*shape[:-1], t.n_bytes // rows)


def bf16_to_f32(a: np.ndarray) -> np.ndarray:
    return (a.reshape(-1).view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def field_value(r: gguf.GGUFReader, key: str):
    f = r.fields.get(key)
    return None if f is None else f.contents()


def tensor_values(readers: list[gguf.GGUFReader], name: str) -> np.ndarray | None:
    for r in readers:
        for t in r.tensors:
            if t.name == name:
                return np.array(t.data)
    return None


def token_map_from_vocab(tokens: list[str], types: list[int]) -> tuple[list[int], int] | None:
    """conversion/deepseek41.py engram_compressed_token_map, with the decode of a byte-level BPE vocab done here
    (the GGUF holds the vocab, not an HF tokenizer dir): normal tokens are byte-decoded, added tokens kept verbatim."""
    try:
        from tokenizers import Regex, normalizers
    except ImportError:
        return None
    byte_dec = {c: b for b, c in _bytes_to_unicode().items()}
    sentinel = ""
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokens)
    for token_id, (tok, typ) in enumerate(zip(tokens, types)):
        if typ == gguf.TokenType.NORMAL:
            text = bytes(byte_dec[c] for c in tok).decode("utf-8", errors="replace")
        else:
            text = tok
        if "�" in text:
            key = tok
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        lookup[token_id] = key_to_new.setdefault(key, len(key_to_new))
    return lookup, len(key_to_new)


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def engram_layout(src: gguf.GGUFReader, readers: list[gguf.GGUFReader], arch: str):
    k = lambda name: field_value(src, f"{arch}.{name}")  # noqa: E731
    layer_ids = [int(x) for x in k("engram.layer_ids")]
    n_emb = [int(x) for x in k("engram.num_embeddings")]
    max_ngram, n_heads, vocab = int(k("engram_max_ngram_size")), int(k("engram_n_heads")), int(k("engram_vocab_size"))
    cvocab = int(k("engram_compressed_vocab_size"))
    n_cols = (max_ngram - 1) * n_heads

    # the converter's own derivation from the layout args, when its imports are available
    primes = mult = None
    try:
        from conversion.deepseek41 import engram_multipliers, engram_primes
        primes = [p for layer in engram_primes(len(layer_ids), max_ngram, n_heads, vocab) for per in layer for p in per]
        mult = [m for row in engram_multipliers(layer_ids, max_ngram, cvocab) for m in row]
    except ImportError as e:
        logger.warning(f"conversion.deepseek41 not importable ({e}); using the packer's stored Engram constants")

    # the packer's stored constants: [layer][head] primes, per-layer offsets, [layer][ngram] multipliers
    st_primes = tensor_values(readers, "engram_primes.weight")
    st_offs = tensor_values(readers, "engram_offsets.weight")
    st_mult = tensor_values(readers, "engram_multipliers.weight")
    if st_primes is not None:
        st_primes = [int(x) for x in st_primes.reshape(-1)]
        if primes is not None and st_primes != primes:
            raise ValueError("stored Engram primes differ from the converter's")
        primes = st_primes
    if st_mult is not None:
        st_mult = [int(x) for x in st_mult.reshape(-1)]
        if mult is not None and st_mult != mult:
            raise ValueError("stored Engram multipliers differ from the converter's")
        mult = st_mult
    if primes is None or mult is None:
        raise ValueError("no Engram constants: neither conversion.deepseek41 nor stored engram_* tensors")
    if len(primes) != len(layer_ids) * n_cols or len(mult) != len(layer_ids) * max_ngram:
        raise ValueError("Engram constants do not match the layout")
    for li in range(len(layer_ids)):
        if sum(primes[li * n_cols:(li + 1) * n_cols]) != n_emb[li]:
            raise ValueError(f"engram layer {layer_ids[li]}: prime sum != num_embeddings")
    offsets = [sum(primes[:i]) for i in range(len(primes))]  # absolute rows in the concatenated table
    if st_offs is not None:
        per_layer = [int(x) for x in st_offs.reshape(-1)]
        base = [sum(n_emb[:i // n_cols]) for i in range(len(per_layer))]
        if [o + b for o, b in zip(per_layer, base)] != offsets:
            raise ValueError("stored Engram offsets differ from the prime layout")

    tokens = [str(t) for t in field_value(src, "tokenizer.ggml.tokens")]
    types = [int(t) for t in field_value(src, "tokenizer.ggml.token_type")]
    token_map = None
    computed = token_map_from_vocab(tokens, types)
    stored = tensor_values(readers, "engram_token_map.weight")
    stored = None if stored is None else [int(x) for x in stored]
    if computed is not None:
        token_map, cv = computed
        if cv != cvocab:
            raise ValueError(f"compressed vocab {cv} != config {cvocab}")
        if stored is not None and stored != token_map:
            bad = sum(a != b for a, b in zip(stored, token_map))
            raise ValueError(f"computed Engram token map differs from the stored one in {bad} ids")
    elif stored is not None:
        logger.warning("tokenizers not importable; using the packer's stored Engram token map")
        token_map = stored
    else:
        raise ValueError("no Engram token map: install tokenizers or keep engram_token_map in the source")
    if len(token_map) != len(tokens) or max(token_map) + 1 != cvocab:
        raise ValueError("Engram token map does not match the vocab")

    return dict(layer_ids=layer_ids, n_emb=n_emb, max_ngram=max_ngram, n_heads=n_heads, head_dim=int(k("engram_head_dim")),
                primes=primes, offsets=offsets, mult=mult, pad=int(k("engram_pad_token_id")), token_map=token_map,
                vocab=vocab, cvocab=cvocab)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", type=Path, help="first shard of the packer's GGUF set")
    ap.add_argument("engram", type=Path, help="the separate Engram GGUF (blk.N.engram_embd.weight)")
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--meta", type=Path, help="a GGUF with #31086's keys (e.g. a llama.cpp conversion's shard 1) to cross-check")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", args.model.name)
    paths = [args.model.with_name(f"{m.group(1)}-{i:05d}-of-{m.group(2)}.gguf") for i in range(1, int(m.group(2)) + 1)] if m else [args.model]
    readers = [gguf.GGUFReader(p) for p in paths]
    src = readers[0]
    arch = field_value(src, "general.architecture")
    if arch != "deepseek41":
        raise ValueError(f"expected arch deepseek41, got {arch}")
    eg = gguf.GGUFReader(args.engram)
    if field_value(eg, f"{arch}.engram.source") != field_value(src, f"{arch}.engram.source"):
        raise ValueError("the Engram file's engram.source does not match the model's")

    lay = engram_layout(src, readers, arch)
    stem = m.group(1) if m else args.model.stem
    w = gguf.GGUFWriter(args.outdir / f"{stem}.gguf", arch, dry_run=args.dry_run)

    # metadata: everything but the GGUF/split bookkeeping and the replaced keys
    for key, f in src.fields.items():
        if key.startswith(("GGUF.", "split.")) or key == "general.architecture" or key.removeprefix(f"{arch}.") in DROP_KEYS:
            continue
        sub = f.types[1] if f.types[0] == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(key, f.contents(), f.types[0], sub)
    k = lambda name: field_value(src, f"{arch}.{name}")  # noqa: E731
    w.add_attention_kv_source_layers([int(x) for x in k("kv_source_layer_ids")])
    w.add_indexer_source_layers([int(x) for x in k("index_source_layer_ids")])
    if k("attention.candidate_source_layer_id") is not None:
        w.add_indexer_candidate_source_layer(int(k("attention.candidate_source_layer_id")))
        w.add_indexer_candidate_top_k_blocks(int(k("attention.candidate_topk_blocks")))
        w.add_indexer_candidate_block_size(int(k("attention.candidate_block_size")))
    w.add_key_value(gguf.Keys.PerLayerEmbedding.LAYERS.format(arch=arch), lay["layer_ids"], gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.UINT32)
    w.add_ple_ngram_size(lay["max_ngram"])
    w.add_ple_heads_per_ngram(lay["n_heads"])
    w.add_embedding_length_per_layer_input(lay["head_dim"])
    w.add_ple_layer_multipliers(lay["mult"])
    w.add_ple_head_offsets(lay["offsets"])
    w.add_ple_head_vocab_sizes(lay["primes"])
    w.add_engram_pad_token_id(lay["pad"])
    w.add_engram_token_map(lay["token_map"])
    w.add_uint32(f"{arch}.engram.vocab_size", lay["vocab"])
    w.add_uint32(f"{arch}.engram.compressed_vocab_size", lay["cvocab"])

    # tensors: one output split per input shard, the Engram table as the last split
    n_copied = 0
    for si, r in enumerate(readers):
        if si > 0:
            w.tensors.append({})
        names = {t.name: t for t in r.tensors}
        for t in r.tensors:
            name = t.name
            if name in HELPER_TENSORS or name.endswith(".engram_k.weight"):
                continue
            if name.endswith(".engram_q.weight"):
                kt = names[name.replace(".engram_q.", ".engram_k.")]
                assert t.tensor_type == kt.tensor_type == gguf.GGMLQuantizationType.BF16
                gate = (bf16_to_f32(t.data) * bf16_to_f32(kt.data)).reshape([int(n) for n in reversed(t.shape)])
                w.add_tensor(name.replace(".engram_q.", ".engram_gate."), gate)
                continue
            if name.endswith(".engram_wkv.weight"):
                name = name.replace(".engram_wkv.", ".engram_kv.")
            w.add_tensor(name, raw_bytes(t), raw_dtype=t.tensor_type)
            n_copied += 1

    parts, ttype = [], None
    by_name = {t.name: t for t in eg.tensors}
    for li, il in enumerate(lay["layer_ids"]):
        t = by_name[f"blk.{il}.engram_embd.weight"]
        if int(t.shape[0]) != lay["head_dim"] or int(t.shape[1]) != lay["n_emb"][li]:
            raise ValueError(f"{t.name}: shape {list(t.shape)} does not match the Engram layout")
        if ttype not in (None, t.tensor_type):
            raise ValueError("Engram tables of different types cannot be concatenated")
        ttype = t.tensor_type
        parts.append(raw_bytes(t))
    w.tensors.append({})
    w.add_tensor(gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD] + ".weight", ConcatRows(parts), raw_dtype=ttype)  # type: ignore[arg-type]

    if args.meta:
        check_meta(w, gguf.GGUFReader(args.meta), arch)

    logger.info(f"{n_copied} tensors copied, Engram table {ttype.name if ttype else '?'} {sum(lay['n_emb'])} rows")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()


def check_meta(w: gguf.GGUFWriter, meta: gguf.GGUFReader, arch: str) -> None:
    """Every #31086 key the reference file has must match ours exactly."""
    ours = w.kv_data[0]
    keys = [kk for kk in meta.fields if kk.startswith(f"{arch}.") and ".dspark." not in kk]
    n = 0
    for kk in keys:
        if kk not in ours:
            if kk != f"{arch}.embedding_length_out":  # unused by deepseek41 (n_embd_out = n_embd)
                logger.warning(f"meta key {kk} is not in the output")
            continue
        a, b = ours[kk].value, meta.fields[kk].contents()
        if isinstance(a, list):
            a = [x for x in a]
        if a != b and not (isinstance(a, float) and abs(a - b) <= 1e-6 * abs(b)):
            raise ValueError(f"{kk}: ours {str(a)[:80]} != meta {str(b)[:80]}")
        if ours[kk].type != meta.fields[kk].types[0]:
            raise ValueError(f"{kk}: type {ours[kk].type.name} != meta {meta.fields[kk].types[0].name}")
        n += 1
    logger.info(f"meta cross-check: {n} keys equal")


if __name__ == "__main__":
    main()
