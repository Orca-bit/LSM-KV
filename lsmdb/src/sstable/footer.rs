use crate::util::coding::{decode_fixed_64, put_fixed_64};
use crate::util::varint::{VarintU64, MAX_VARINT_LEN_U64};
use crate::{Error, Result};

const TABLE_MAGIC_NUMBER: u64 = 0xdb4775248b80fb57;
// 1byte compression type + 4bytes cyc
pub(crate) const BLOCK_TRAILER_SIZE: usize = 5;
const MAX_BLOCK_HANDLE_ENCODE_LENGTH: usize = 2 * MAX_VARINT_LEN_U64;
// Encoded length of a Footer.  Note that the serialization of a
// Footer will always occupy exactly this many bytes.  It consists
// of two block handles and a magic number. 48 Bytes
pub(crate) const FOOTER_ENCODED_LENGTH: usize = 2 * MAX_BLOCK_HANDLE_ENCODE_LENGTH + 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockHandle {
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

impl BlockHandle {
    pub fn new(offset: u64, size: u64) -> Self {
        Self { offset, size }
    }

    #[inline]
    pub fn set_offset(&mut self, offset: u64) {
        self.offset = offset;
    }

    #[inline]
    pub fn set_size(&mut self, size: u64) {
        self.size = size;
    }

    #[inline]
    pub fn encoded_to(&self, dst: &mut Vec<u8>) {
        VarintU64::put_varint(dst, self.offset);
        VarintU64::put_varint(dst, self.size);
    }

    #[inline]
    pub fn encoded(&self) -> Vec<u8> {
        let mut res = vec![];
        self.encoded_to(&mut res);
        res
    }

    #[inline]
    pub fn decode_from(src: &[u8]) -> Result<(Self, usize)> {
        if let Some((offset, n)) = VarintU64::read(src) {
            if let Some((size, m)) = VarintU64::read(&src[n..]) {
                Ok((Self::new(offset, size), n + m))
            } else {
                Err(Error::Corruption("bad block handle".to_owned()))
            }
        } else {
            Err(Error::Corruption("bad block handle".to_owned()))
        }
    }
}

#[derive(Debug)]
pub struct Footer {
    pub(crate) meta_index_handle: BlockHandle,
    pub(crate) index_handle: BlockHandle,
}

impl Footer {
    pub fn new(meta_index_handle: BlockHandle, index_handle: BlockHandle) -> Self {
        Self {
            meta_index_handle,
            index_handle,
        }
    }

    pub fn decode_from(src: &[u8]) -> Result<(Self, usize)> {
        let magic = decode_fixed_64(&src[FOOTER_ENCODED_LENGTH - 8..]);
        if magic != TABLE_MAGIC_NUMBER {
            return Err(Error::Corruption(
                "not an sstable (bad magic number)".to_owned(),
            ));
        }
        let (meta_index_handle, n) = BlockHandle::decode_from(src)?;
        let (index_handle, m) = BlockHandle::decode_from(&src[n..])?;
        Ok((
            Self {
                meta_index_handle,
                index_handle,
            },
            n + m,
        ))
    }

    pub fn encoded(&self) -> Vec<u8> {
        let mut res = vec![];
        self.meta_index_handle.encoded_to(&mut res);
        self.index_handle.encoded_to(&mut res);
        res.resize(MAX_BLOCK_HANDLE_ENCODE_LENGTH * 2, 0);
        put_fixed_64(&mut res, TABLE_MAGIC_NUMBER);
        assert_eq!(
            res.len(),
            FOOTER_ENCODED_LENGTH,
            "[footer] the len of encoded footer is {}, expect {}",
            res.len(),
            FOOTER_ENCODED_LENGTH
        );
        res
    }
}

#[cfg(test)]
mod test_footer {
    use super::*;

    #[test]
    fn test_footer_corruption() {
        let footer = Footer::new(BlockHandle::new(300, 100), BlockHandle::new(401, 1000));
        let mut encoded = footer.encoded();
        let last = encoded.last_mut().unwrap();
        *last += 1;
        let r1 = Footer::decode_from(&encoded);
        assert!(r1.is_err());
        let e1 = r1.unwrap_err();
        assert_eq!(
            e1.to_string(),
            "data corruption: not an sstable (bad magic number)"
        );
    }

    #[test]
    fn test_encode_decode() {
        let footer = Footer::new(BlockHandle::new(300, 100), BlockHandle::new(401, 1000));
        let encoded = footer.encoded();
        let (footer, _) = Footer::decode_from(&encoded).expect("footer decoding should work");
        assert_eq!(footer.index_handle, BlockHandle::new(401, 1000));
        assert_eq!(footer.meta_index_handle, BlockHandle::new(300, 100));
    }
}
