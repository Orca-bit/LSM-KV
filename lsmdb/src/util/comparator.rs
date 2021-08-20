use std::cmp::Ordering;

pub trait Comparator: Send + Sync + Clone + Default {
    fn compare(&self, a: &[u8], b: &[u8]) -> Ordering;
    fn name(&self) -> &str;
    fn separator(&self, a: &[u8], b: &[u8]) -> Vec<u8>;
    fn successor(&self, key: &[u8]) -> Vec<u8>;
}

#[derive(Default, Clone, Copy)]
pub struct BytewiseComparator {}

impl Comparator for BytewiseComparator {
    #[inline]
    fn compare(&self, a: &[u8], b: &[u8]) -> Ordering {
        a.cmp(b)
    }

    #[inline]
    fn name(&self) -> &str {
        "leveldb.BytewiseComparator"
    }

    #[inline]
    fn separator(&self, a: &[u8], b: &[u8]) -> Vec<u8> {
        let min_size = a.len().min(b.len());
        let mut diff_index = 0;
        while diff_index < min_size && a[diff_index] == b[diff_index] {
            diff_index += 1;
        }
        if diff_index >= min_size {
            // one is the prefix of the other
        } else {
            let last = a[diff_index];
            if last != 0xff && last + 1 < b[diff_index] {
                let mut res = vec![0; diff_index + 1];
                res[0..=diff_index].copy_from_slice(&a[0..=diff_index]);
                *(res.last_mut().unwrap()) += 1;
                return res;
            }
        }
        a.to_owned()
    }

    #[inline]
    fn successor(&self, key: &[u8]) -> Vec<u8> {
        // Find first character that can be incremented
        for i in 0..key.len() {
            let byte = key[i];
            if byte != 0xff {
                let mut res: Vec<u8> = vec![0; i + 1];
                res[0..=i].copy_from_slice(&key[0..=i]);
                *(res.last_mut().unwrap()) += 1;
                return res;
            }
        }
        key.to_owned()
    }
}
