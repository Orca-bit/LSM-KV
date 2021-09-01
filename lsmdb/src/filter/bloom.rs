use crate::filter::FilterPolicy;
use crate::util::hash::hash;

pub struct BloomFilter {
    // the hash count for a key
    k: usize,
    bits_per_key: usize,
}

impl BloomFilter {
    pub fn new(bits_per_key: usize) -> Self {
        // 0.69 =~ ln(2) and we intentionally round down to reduce probing cost a little bit
        let mut k = bits_per_key as f32 * 0.69;
        if k > 30_f32 {
            k = 30_f32;
        } else if k < 1_f32 {
            k = 1_f32;
        }
        Self {
            k: k as usize,
            bits_per_key,
        }
    }

    fn bloom_hash(data: &[u8]) -> u32 {
        hash(data, 0xc6a4a793)
    }
}

impl FilterPolicy for BloomFilter {
    fn name(&self) -> &str {
        "BuiltinBloomFilter"
    }

    fn may_contain(&self, filter: &[u8], key: &[u8]) -> bool {
        let n = filter.len() - 1;
        if filter.is_empty() || n < 1 {
            return false;
        }
        let bits = n << 3;
        let k = filter[n];
        if k > 30 {
            return true;
        }
        let mut h = Self::bloom_hash(key);
        let delta = (h >> 17) | (h << 15);
        for _ in 0..k {
            let bit_pos = h % (bits as u32);
            if (filter[(bit_pos >> 3) as usize] & (1 << (bit_pos % 8))) == 0 {
                return false;
            }
            h = h.wrapping_add(delta);
        }
        true
    }

    fn create_filter(&self, keys: &[Vec<u8>]) -> Vec<u8> {
        let mut bits = keys.len() * self.bits_per_key;
        if bits < 64 {
            bits = 64;
        }
        let bytes = (bits + 7) >> 3;
        bits = bytes << 3;
        let mut res = vec![0; bytes + 1];
        res[bytes] = self.k as u8;

        for key in keys {
            let mut h = Self::bloom_hash(key);
            let delta = (h >> 17) | (h << 15);
            for _ in 0..self.k {
                let bit_pos = h % (bits as u32);
                res[(bit_pos >> 3) as usize] |= 1 << (bit_pos % 8);
                h = h.wrapping_add(delta);
            }
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::coding::encode_fixed_32;

    struct Harness {
        policy: Box<dyn FilterPolicy>,
        filter: Vec<u8>,
        keys: Vec<Vec<u8>>,
    }

    impl Harness {
        fn new() -> Self {
            Self {
                policy: Box::new(BloomFilter::new(10)),
                filter: vec![],
                keys: vec![],
            }
        }

        fn add_key(&mut self, key: Vec<u8>) {
            self.keys.push(key);
        }

        fn add_num(&mut self, num: u32) {
            let mut k: Vec<u8> = vec![0; 4];
            encode_fixed_32(k.as_mut_slice(), num);
            self.add_key(k);
        }

        fn filter_len(&self) -> usize {
            self.filter.len()
        }

        fn assert_or_return(&self, key: &[u8], want: bool, assert: bool) -> bool {
            let got = (&self).policy.may_contain(self.filter.as_slice(), key);
            if assert {
                assert_eq!(got, want);
            };
            got
        }

        fn assert_num(&self, key: u32, want: bool, silent: bool) -> bool {
            let mut k: Vec<u8> = vec![0; 4];
            encode_fixed_32(k.as_mut_slice(), key);
            self.assert_or_return(k.as_slice(), want, !silent)
        }

        fn build(&mut self) {
            self.filter = (&self).policy.create_filter(self.keys.as_slice());
        }

        fn reset(&mut self) {
            self.filter.clear();
            self.keys.clear();
        }
    }

    fn next_n(n: u32) -> u32 {
        match n {
            _ if n < 10 => {
                return n + 1;
            }
            _ if n < 100 => {
                return n + 10;
            }
            _ if n < 1000 => {
                return n + 100;
            }
            _ => return n + 1000,
        };
    }

    #[test]
    fn test_bloom_filter_empty() {
        let mut h = Harness::new();
        h.build();
        h.assert_or_return("hello".as_bytes(), false, true);
        h.assert_or_return("world".as_bytes(), false, true);
    }

    #[test]
    fn test_bloom_filter_small() {
        let mut h = Harness::new();
        h.add_key(Vec::from("hello"));
        h.add_key(Vec::from("world"));
        h.build();
        h.assert_or_return("hello".as_bytes(), true, true);
        h.assert_or_return("world".as_bytes(), true, true);
        h.assert_or_return("x".as_bytes(), false, true);
        h.assert_or_return("foo".as_bytes(), false, true);
    }

    #[test]
    fn test_bloom_filter_varying_lengths() {
        let mut h = Harness::new();
        let mut n: u32 = 1;
        let mut mediocre = 0;
        let mut good = 0;
        while n < 10000 {
            h.reset();
            for i in 0..n {
                h.add_num(i);
            }
            h.build();
            let got = h.filter_len();
            let want = (n * 10 / 8) + 40;
            assert_eq!(
                got as u32 <= want,
                true,
                "filter len test failed, '{}' > '{}'",
                got,
                want
            );
            for i in 0..n {
                h.assert_num(i, true, false);
            }

            let mut rate: f32 = 0.0;
            for i in 0..n {
                if h.assert_num(i + 1000000000, true, true) {
                    rate += 1.0;
                }
            }
            rate /= 10000.0;
            assert!(
                rate <= 0.02,
                "false positive rate is more than 2%%, got {}, at len {}",
                rate,
                n
            );
            if rate > 0.0125 {
                mediocre += 1;
            } else {
                good += 1;
            }
            n = next_n(n);
        }
        assert!(
            mediocre <= good / 5,
            "mediocre false positive rate is more than expected"
        );
    }
}
