#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// BPK1, little-endian.
//   u32 magic 'BPK1' (0x314B5042)
//   u32 version = 1
//   u32 tensor_count
//   repeated:
//     u32 name_len, name bytes (no NUL)
//     u32 ndim, u32 dims[ndim]
//     u64 payload_nbytes
//     payload (BF16 little-endian bits)
//     zero pad to an 8-byte file offset

struct PackedTensor {
    std::string name;
    std::vector<uint32_t> dims;
    std::vector<uint16_t> bf16;
};

namespace pack_detail {

inline uint32_t read_u32(std::ifstream& in) {
    unsigned char b[4];
    in.read(reinterpret_cast<char*>(b), 4);
    if (!in)
        throw std::runtime_error("pack: truncated u32");
    return uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
}

inline uint64_t read_u64(std::ifstream& in) {
    unsigned char b[8];
    in.read(reinterpret_cast<char*>(b), 8);
    if (!in)
        throw std::runtime_error("pack: truncated u64");
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i)
        v |= uint64_t(b[i]) << (8 * i);
    return v;
}

} // namespace pack_detail

inline std::vector<PackedTensor> load_pack(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("pack: cannot open " + path);
    const uint32_t magic = pack_detail::read_u32(in);
    const uint32_t version = pack_detail::read_u32(in);
    const uint32_t count = pack_detail::read_u32(in);
    if (magic != 0x314B5042u || version != 1u) {
        throw std::runtime_error("pack: bad header in " + path);
    }
    std::vector<PackedTensor> out;
    out.reserve(count);
    std::uint64_t offset = 12;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t name_len = pack_detail::read_u32(in);
        offset += 4;
        PackedTensor t;
        t.name.resize(name_len);
        if (name_len) {
            in.read(t.name.data(), name_len);
            if (!in)
                throw std::runtime_error("pack: truncated name");
        }
        offset += name_len;
        const uint32_t ndim = pack_detail::read_u32(in);
        offset += 4;
        if (ndim > 4)
            throw std::runtime_error("pack: ndim too large");
        t.dims.resize(ndim);
        for (uint32_t d = 0; d < ndim; ++d) {
            t.dims[d] = pack_detail::read_u32(in);
            offset += 4;
        }
        const uint64_t nbytes = pack_detail::read_u64(in);
        offset += 8;
        if (nbytes % 2u)
            throw std::runtime_error("pack: odd bf16 payload");
        t.bf16.resize(nbytes / 2u);
        if (nbytes) {
            in.read(reinterpret_cast<char*>(t.bf16.data()), std::streamsize(nbytes));
            if (!in)
                throw std::runtime_error("pack: truncated payload " + t.name);
        }
        offset += nbytes;
        const uint64_t pad = (8u - (offset & 7u)) & 7u;
        if (pad) {
            in.seekg(std::streamoff(pad), std::ios::cur);
            if (!in)
                throw std::runtime_error("pack: truncated pad");
            offset += pad;
        }
        out.push_back(std::move(t));
    }
    return out;
}
