use std::mem::transmute;
use std::ptr::copy_nonoverlapping;

/// Encodes `value` in little-endian and puts it in the first 4-bytes of `dst`.
///
/// # Panics
///
/// Panics if `dst.len()` is less than 4.
pub fn encode_fixed_32(dst: &mut [u8], value: u32) {
    assert!(
        dst.len() >= 4,
        "the length of 'dst' must be at least 4 for a u32, but got {}",
        dst.len()
    );
    unsafe {
        let bytes = transmute::<u32, [u8; 4]>(value.to_le());
        copy_nonoverlapping(bytes.as_ptr(), dst.as_mut_ptr(), 4);
    }
}

/// Encodes `value` in little-endian and puts in the first 8-bytes of `dst`.
///
/// # Panics
///
/// Panics if `dst.len()` is less than 8.
pub fn encode_fixed_64(dst: &mut [u8], value: u64) {
    assert!(
        dst.len() >= 8,
        "the length of 'dst' must be at least 8 for a u64, but got {}",
        dst.len()
    );
    unsafe {
        let bytes = transmute::<u64, [u8; 8]>(value.to_le());
        copy_nonoverlapping(bytes.as_ptr(), dst.as_mut_ptr(), 8);
    }
}

/// Decodes the first 4-bytes of `src` in little-endian and returns the decoded value.
///
/// If the length of the given `src` is larger than 4, only use `src[0..4]`
pub fn decode_fixed_32(src: &[u8]) -> u32 {
    let mut data: u32 = 0;
    if src.len() >= 4 {
        unsafe {
            copy_nonoverlapping(src.as_ptr(), &mut data as *mut u32 as *mut u8, 4);
        }
    } else {
        for (i, b) in src.iter().enumerate() {
            data += (u32::from(*b)) << (i * 8);
        }
    }
    data.to_le()
}

/// Decodes the first 8-bytes of `src` in little-endian and returns the decoded value.
///
/// If the length of the given `src` is larger than 8, only use `src[0..8]`
pub fn decode_fixed_64(src: &[u8]) -> u64 {
    let mut data: u64 = 0;
    if src.len() >= 8 {
        unsafe {
            copy_nonoverlapping(src.as_ptr(), &mut data as *mut u64 as *mut u8, 8);
        }
    } else {
        for (i, b) in src.iter().enumerate() {
            data += (u64::from(*b)) << (i * 8);
        }
    }
    data.to_le()
}

/// Encodes the given u32 to bytes and concatenates the result to `dst`
pub fn put_fixed_32(dst: &mut Vec<u8>, value: u32) {
    let mut buf = [0u8; 4];
    encode_fixed_32(&mut buf[..], value);
    dst.extend_from_slice(&buf);
}

/// Encodes the given u64 to bytes and concatenates the result to `dst`
pub fn put_fixed_64(dst: &mut Vec<u8>, value: u64) {
    let mut buf = [0u8; 8];
    encode_fixed_64(&mut buf[..], value);
    dst.extend_from_slice(&buf);
}
