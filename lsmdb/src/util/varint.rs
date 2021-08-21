pub const MAX_VARINT_LEN_U32: usize = 5;
pub const MAX_VARINT_LEN_U64: usize = 10;

pub struct VarintU32;
pub struct VarintU64;

macro_rules! impl_varint {
    ($type:ty, $uint: ty) => {
        impl $type {
            /// Encodes a uint into given vec and returns the number of bytes written.
            /// Using little endian style.
            /// See Varint in https://developers.google.com/protocol-buffers/docs/encoding#varints
            ///
            /// # Panic
            ///
            /// Panic when `dst` length is not enough
            pub fn write(dst: &mut [u8], mut n: $uint) -> usize {
                let mut i = 0;
                while n >= 0b1000_0000 {
                    dst[i] = (n as u8) | 0b1000_0000;
                    n >>= 7;
                    i += 1;
                }
                dst[i] = n as u8;
                i + 1
            }

            /// Decodes a uint(32 or 64) from given bytes and returns that value and the
            /// number of bytes read ( > 0).
            /// If an error or overflow occurred, returns `None`
            pub fn read(src: &[u8]) -> Option<($uint, usize)> {
                let mut n: $uint = 0;
                let mut shift: u32 = 0;
                for (i, &b) in src.iter().enumerate() {
                    if b < 0b1000_0000 {
                        return match (<$uint>::from(b)).checked_shl(shift) {
                            None => None,
                            Some(b) => Some((n | b, (i + 1) as usize)),
                        };
                    }
                    match ((<$uint>::from(b)) & 0b0111_1111).checked_shl(shift) {
                        None => return None,
                        Some(b) => n |= b,
                    }
                    shift += 7;
                }
                None
            }

            /// Append `n` as varint bytes into the dst.
            /// Returns the bytes written.
            pub fn put_varint(dst: &mut Vec<u8>, mut n: $uint) -> usize {
                let mut i = 0;
                while n >= 0b1000_0000 {
                    dst.push((n as u8) | 0b1000_0000);
                    n >>= 7;
                    i += 1;
                }
                dst.push(n as u8);
                i + 1
            }

            /// Encodes the slice `src` into the `dst` as varint length prefixed
            pub fn put_varint_prefixed_slice(dst: &mut Vec<u8>, src: &[u8]) {
                if !src.is_empty() {
                    Self::put_varint(dst, src.len() as $uint);
                    dst.extend_from_slice(src);
                }
            }

            /// Decodes the varint-length-prefixed slice from `src, and advance `src`
            pub fn get_varint_prefixed_slice<'a>(src: &mut &'a [u8]) -> Option<&'a [u8]> {
                Self::read(src).and_then(|(len, n)| {
                    let read_len = len as usize + n;
                    if read_len > src.len() {
                        return None;
                    }
                    let res = &src[n..read_len];
                    *src = &src[read_len..];
                    Some(res)
                })
            }

            /// Decodes a u64 from given bytes and returns that value and the
            /// number of bytes read ( > 0).If an error occurred, the value is 0
            /// and the number of bytes n is <= 0 meaning:
            ///
            ///  n == 0:buf too small
            ///  n  < 0: value larger than 64 bits (overflow)
            ///          and -n is the number of bytes read
            ///
            pub fn common_read(src: &[u8]) -> ($uint, isize) {
                let mut n: $uint = 0;
                let mut shift: u32 = 0;
                for (i, &b) in src.iter().enumerate() {
                    if b < 0b1000_0000 {
                        return match (<$uint>::from(b)).checked_shl(shift) {
                            None => (0, -(i as isize + 1)),
                            Some(b) => (n | b, (i + 1) as isize),
                        };
                    }
                    match ((<$uint>::from(b)) & 0b0111_1111).checked_shl(shift) {
                        None => return (0, -(i as isize)),
                        Some(b) => n |= b,
                    }
                    shift += 7;
                }
                (0, 0)
            }

            /// Decodes a uint from the give slice , and advance the given slice
            pub fn drain_read(src: &mut &[u8]) -> Option<$uint> {
                <$type>::read(src).and_then(|(v, n)| {
                    *src = &src[n..];
                    Some(v)
                })
            }
        }
    };
}

impl_varint!(VarintU32, u32);
impl_varint!(VarintU64, u64);