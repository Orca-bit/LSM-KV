use super::coding::decode_fixed_32;

#[allow(clippy::many_single_char_names)]
pub fn hash(data: &[u8], seed: u32) -> u32 {
    // Similar to murmur hash
    let n = data.len();
    let m: u32 = 0xc6a4a793;
    let mut h = seed ^ (m.wrapping_mul(n as u32));

    // Pick up four bytes at a time
    let mut i = 0;
    while i + 4 <= n {
        let w = decode_fixed_32(&data[i..]);
        i += 4;
        h = h.wrapping_add(w);
        h = h.wrapping_mul(m);
        h ^= h >> 16;
    }

    // Pick up remaining bytes
    let diff = n - i;
    if diff >= 3 {
        h += (u32::from(data[i + 2])) << 16
    };
    if diff >= 2 {
        h += (u32::from(data[i + 1])) << 8
    };
    if diff >= 1 {
        h += u32::from(data[i]);
        h = h.wrapping_mul(m);
        h ^= h >> 24;
    }
    h
}
