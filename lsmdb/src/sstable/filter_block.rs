use crate::filter::FilterPolicy;
use crate::util::coding::{decode_fixed_32, put_fixed_32};
use std::sync::Arc;

const FILTER_BASE_LG: usize = 11;
const FILTER_BASE: usize = 1 << FILTER_BASE_LG;
const FILTER_META_LENGTH: usize = std::mem::size_of::<u32>() + 1;
const FILTER_OFFSET_LEN: usize = std::mem::size_of::<u32>();

pub struct FilterBlockBuilder {
    policy: Arc<dyn FilterPolicy>,
    keys: Vec<Vec<u8>>,
    //
    // |----- filter data -----|----- filter offsets ----|--- filter offsets len ---|--- BASE_LG ---|
    //                                   num * 4 bytes              4 bytes               1 byte
    data: Vec<u8>,
    filter_offsets: Vec<u32>,
}

impl FilterBlockBuilder {
    pub fn new(policy: Arc<dyn FilterPolicy>) -> Self {
        Self {
            policy,
            keys: vec![],
            data: vec![],
            filter_offsets: vec![],
        }
    }

    pub fn add_key(&mut self, key: &[u8]) {
        self.keys.push(Vec::from(key));
    }

    pub fn start_block(&mut self, block_offset: u64) {
        let filter_index = block_offset / FILTER_BASE as u64;
        let filters_len = self.filter_offsets.len() as u64;
        assert!(
            filter_index >= filters_len,
            "[filter block builder] the filter block index {} should larger than built filters {}",
            filter_index,
            filters_len,
        );
        while filter_index > self.filter_offsets.len() as u64 {
            self.generate_filter();
        }
    }

    pub fn finish(&mut self) -> &[u8] {
        if !self.keys.is_empty() {
            self.generate_filter();
        }
        for offset in self.filter_offsets.iter() {
            put_fixed_32(&mut self.data, *offset);
        }
        put_fixed_32(&mut self.data, self.filter_offsets.len() as u32);
        self.data.push(FILTER_BASE_LG as u8);
        &self.data
    }

    fn generate_filter(&mut self) {
        self.filter_offsets.push(self.data.len() as u32);
        if !self.keys.is_empty() {
            let filter = self.policy.create_filter(&self.keys);
            self.data.extend(filter);
            self.keys.clear();
        }
    }
}

pub struct FilterBlockReader {
    policy: Arc<dyn FilterPolicy>,
    data: Vec<u8>,
    num: usize,
    base_lg: usize,
}

impl FilterBlockReader {
    pub fn new(policy: Arc<dyn FilterPolicy>, mut filter_block: Vec<u8>) -> Self {
        let mut r = Self {
            policy,
            data: vec![],
            num: 0,
            base_lg: 0,
        };
        let n = filter_block.len();
        if n < FILTER_META_LENGTH {
            return r;
        }
        r.num = decode_fixed_32(&filter_block[n - FILTER_META_LENGTH..n - 1]) as usize;
        if r.num * FILTER_OFFSET_LEN + FILTER_META_LENGTH > n {
            return r;
        }
        r.base_lg = filter_block[n - 1] as usize;
        filter_block.truncate(n - FILTER_META_LENGTH);
        r.data = filter_block;
        r
    }

    pub fn key_may_match(&self, block_offset: u64, key: &[u8]) -> bool {
        // a >> b == a / (1 << b)
        let i = block_offset as usize >> self.base_lg;
        if i < self.num {
            let (filter, offsets) = &self
                .data
                .split_at(self.data.len() - self.num * FILTER_OFFSET_LEN);
            let start =
                decode_fixed_32(&offsets[i * FILTER_OFFSET_LEN..(i + 1) * FILTER_OFFSET_LEN])
                    as usize;
            let end = if i + 1 >= self.num {
                filter.len()
            } else {
                decode_fixed_32(&offsets[(i + 1) * FILTER_OFFSET_LEN..(i + 2) * FILTER_OFFSET_LEN])
                    as usize
            };
            let filter = &self.data[start..end];
            return self.policy.may_contain(filter, key);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::FilterPolicy;
    use crate::util::hash::hash;

    struct TestHashFilter {}

    impl FilterPolicy for TestHashFilter {
        fn name(&self) -> &str {
            "TestHashFilter"
        }

        fn may_contain(&self, filter: &[u8], key: &[u8]) -> bool {
            let h = hash(key, 1);
            let mut i = 0;
            while i + 4 <= filter.len() {
                if h == decode_fixed_32(&filter[i..i + 4]) {
                    return true;
                }
                i += 4;
            }
            false
        }

        fn create_filter(&self, keys: &[Vec<u8>]) -> Vec<u8> {
            let mut f = vec![];
            for i in 0..keys.len() {
                let h = hash(keys[i].as_slice(), 1);
                put_fixed_32(&mut f, h);
            }
            f
        }
    }

    fn new_test_builder() -> FilterBlockBuilder {
        FilterBlockBuilder::new(Arc::new(TestHashFilter {}))
    }
    fn new_test_reader(block: Vec<u8>) -> FilterBlockReader {
        FilterBlockReader::new(Arc::new(TestHashFilter {}), block)
    }

    #[test]
    fn test_empty_builder() {
        let mut b = new_test_builder();
        let block = b.finish();
        assert_eq!(&[0, 0, 0, 0, FILTER_BASE_LG as u8], block);
        let r = new_test_reader(Vec::from(block));
        assert_eq!(r.key_may_match(0, "foo".as_bytes()), true);
        assert_eq!(r.key_may_match(10000, "foo".as_bytes()), true);
    }

    #[test]
    fn test_single_chunk() {
        let mut b = new_test_builder();
        b.start_block(100);
        b.add_key("foo".as_bytes());
        b.add_key("bar".as_bytes());
        b.add_key("box".as_bytes());
        b.start_block(200);
        b.add_key("box".as_bytes());
        b.start_block(300);
        b.add_key("hello".as_bytes());
        let block = b.finish();
        let r = new_test_reader(Vec::from(block));
        assert_eq!(r.key_may_match(100, "foo".as_bytes()), true);
        assert_eq!(r.key_may_match(100, "bar".as_bytes()), true);
        assert_eq!(r.key_may_match(100, "box".as_bytes()), true);
        assert_eq!(r.key_may_match(100, "hello".as_bytes()), true);
        assert_eq!(r.key_may_match(100, "foo".as_bytes()), true);
        assert_eq!(r.key_may_match(100, "missing".as_bytes()), false);
        assert_eq!(r.key_may_match(100, "other".as_bytes()), false);
    }

    #[test]
    fn test_multiple_chunk() {
        let mut b = new_test_builder();
        // first filter
        b.start_block(0);
        b.add_key("foo".as_bytes());
        b.start_block(2000);
        b.add_key("bar".as_bytes());

        // second filter
        b.start_block(3100);
        b.add_key("box".as_bytes());

        // third filter is empty

        // last filter
        b.start_block(9000);
        b.add_key("box".as_bytes());
        b.add_key("hello".as_bytes());
        let block = b.finish();
        let r = new_test_reader(Vec::from(block));

        // check first filter
        assert_eq!(r.key_may_match(0, "foo".as_bytes()), true);
        assert_eq!(r.key_may_match(2000, "bar".as_bytes()), true);
        assert_eq!(r.key_may_match(0, "box".as_bytes()), false);
        assert_eq!(r.key_may_match(0, "hello".as_bytes()), false);
        // check second filter
        assert_eq!(r.key_may_match(3100, "box".as_bytes()), true);
        assert_eq!(r.key_may_match(3100, "foo".as_bytes()), false);
        assert_eq!(r.key_may_match(3100, "bar".as_bytes()), false);
        assert_eq!(r.key_may_match(3100, "hello".as_bytes()), false);
        // check third filter (empty)
        assert_eq!(r.key_may_match(4100, "box".as_bytes()), false);
        assert_eq!(r.key_may_match(4100, "foo".as_bytes()), false);
        assert_eq!(r.key_may_match(4100, "bar".as_bytes()), false);
        assert_eq!(r.key_may_match(4100, "hello".as_bytes()), false);
        // check last filter
        assert_eq!(r.key_may_match(9000, "box".as_bytes()), true);
        assert_eq!(r.key_may_match(9000, "foo".as_bytes()), false);
        assert_eq!(r.key_may_match(9000, "bar".as_bytes()), false);
        assert_eq!(r.key_may_match(9000, "hello".as_bytes()), true);
    }
}
