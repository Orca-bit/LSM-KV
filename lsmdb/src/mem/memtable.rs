use crate::db::format::{InternalKeyComparator, LookupKey, ValueType, INTERNAL_KEY_TAIL};
use crate::mem::arena::OffsetArena;
use crate::mem::inlineskiplist::{InlineSkipList, InlineSkiplistIterator};
use crate::util::coding::{decode_fixed_64, put_fixed_64};
use crate::util::comparator::Comparator;
use crate::util::varint::VarintU32;
use crate::{Error, Iterator, Result};
use std::cmp::Ordering;

#[derive(Clone, Default)]
pub struct KeyComparator<C: Comparator> {
    cmp: InternalKeyComparator<C>,
}

impl<C> Comparator for KeyComparator<C>
where
    C: Comparator,
{
    fn compare(&self, mut a: &[u8], mut b: &[u8]) -> Ordering {
        let a = extract_varint32_encoded_slice(&mut a);
        let b = extract_varint32_encoded_slice(&mut b);
        if a.is_empty() || b.is_empty() {
            a.cmp(&b)
        } else {
            self.cmp.compare(a, b)
        }
    }

    fn name(&self) -> &str {
        self.cmp.name()
    }

    fn separator(&self, mut a: &[u8], mut b: &[u8]) -> Vec<u8> {
        let a = extract_varint32_encoded_slice(&mut a);
        let b = extract_varint32_encoded_slice(&mut b);
        self.cmp.separator(a, b)
    }

    fn successor(&self, mut key: &[u8]) -> Vec<u8> {
        let a = extract_varint32_encoded_slice(&mut key);
        self.cmp.successor(a)
    }
}

pub struct MemTable<C: Comparator> {
    cmp: KeyComparator<C>,
    table: InlineSkipList<KeyComparator<C>, OffsetArena>,
}

impl<C> MemTable<C>
where
    C: Comparator,
{
    pub fn new(max_mem_size: usize, cmp: InternalKeyComparator<C>) -> Self {
        let arena = OffsetArena::new_with_capacity(max_mem_size);
        let cmp = KeyComparator { cmp };
        let table = InlineSkipList::new(cmp.clone(), arena);
        Self { cmp, table }
    }

    #[inline]
    pub fn approximate_memory_usage(&self) -> usize {
        self.table.total_size()
    }

    pub fn iter(&self) -> MemTableIterator<C> {
        MemTableIterator::new(self.table.clone())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.table.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.table.len() == 0
    }

    /// Add an entry into memtable that maps key to value at the
    /// specified sequence number and with the specified type.
    /// Typically value will be empty if the type is `Deletion`.
    ///
    /// The 'key' and 'value' will be bundled together into an 'entry':
    ///
    /// ```text
    ///   +=================================+
    ///   |       format of the entry       |
    ///   +=================================+
    ///   | varint32 of internal key length |
    ///   +---------------------------------+ ---------------
    ///   | user key bytes                  |
    ///   +---------------------------------+   internal key
    ///   | sequence (7)       |   type (1) |
    ///   +---------------------------------+ ---------------
    ///   | varint32 of value length        |
    ///   +---------------------------------+
    ///   | value bytes                     |
    ///   +---------------------------------+
    /// ```
    ///
    pub fn add(&self, seq_number: u64, val_type: ValueType, key: &[u8], value: &[u8]) {
        let key_size = key.len();
        let internal_key_size = key_size + INTERNAL_KEY_TAIL;
        let mut buf = vec![];
        VarintU32::put_varint(&mut buf, internal_key_size as u32);
        buf.extend_from_slice(key);
        put_fixed_64(
            &mut buf,
            (seq_number << INTERNAL_KEY_TAIL) | val_type as u64,
        );
        VarintU32::put_varint_prefixed_slice(&mut buf, value);
        self.table.put(buf);
    }

    pub fn get(&self, key: &LookupKey) -> Option<Result<Vec<u8>>> {
        let mk = key.mem_key();
        let mut iter = InlineSkiplistIterator::new(self.table.clone());
        iter.seek(mk);
        if iter.valid() {
            let mut e = iter.key();
            let ikey = extract_varint32_encoded_slice(&mut e);
            let key_size = ikey.len();
            // only check the user key here
            match self
                .cmp
                .cmp
                .user_comparator
                .compare(&ikey[..key_size - INTERNAL_KEY_TAIL], key.user_key())
            {
                Ordering::Equal => {
                    let tag = decode_fixed_64(&ikey[key_size - INTERNAL_KEY_TAIL..]);
                    match ValueType::from(tag & 0xff_u64) {
                        ValueType::Value => {
                            return Some(Ok(extract_varint32_encoded_slice(&mut e).to_vec()))
                        }
                        ValueType::Deletion => return Some(Err(Error::NotFound(None))),
                        ValueType::Unknown => { /* fallback to None*/ }
                    }
                }
                _ => return None,
            }
        }
        None
    }
}

pub struct MemTableIterator<C: Comparator> {
    iter: InlineSkiplistIterator<KeyComparator<C>, OffsetArena>,
    // Tmp buffer for encoding `InternalKey` to `LookupKey` when call `seek`
    tmp: Vec<u8>,
}

impl<C: Comparator> MemTableIterator<C> {
    pub fn new(table: InlineSkipList<KeyComparator<C>, OffsetArena>) -> Self {
        let iter = InlineSkiplistIterator::new(table);
        Self { iter, tmp: vec![] }
    }
}

impl<C: Comparator> Iterator for MemTableIterator<C> {
    fn valid(&self) -> bool {
        self.iter.valid()
    }

    fn seek_to_first(&mut self) {
        self.iter.seek_to_first()
    }

    fn seek_to_last(&mut self) {
        self.iter.seek_to_last()
    }

    // target should be an encoded `LookupKey`
    fn seek(&mut self, target: &[u8]) {
        self.tmp.clear();
        VarintU32::put_varint_prefixed_slice(&mut self.tmp, target);
        self.iter.seek(&self.tmp)
    }

    fn next(&mut self) {
        self.iter.next()
    }

    fn prev(&mut self) {
        self.iter.prev()
    }

    // Returns the internal key
    fn key(&self) -> &[u8] {
        let mut key = self.iter.key();
        extract_varint32_encoded_slice(&mut key)
    }

    // Returns the Slice represents the value
    fn value(&self) -> &[u8] {
        let mut key = self.iter.key();
        extract_varint32_encoded_slice(&mut key);
        extract_varint32_encoded_slice(&mut key)
    }

    fn status(&mut self) -> Result<()> {
        Ok(())
    }
}

// Decodes the length (varint u32) from `src` and advances it.
fn extract_varint32_encoded_slice<'a>(src: &mut &'a [u8]) -> &'a [u8] {
    if src.is_empty() {
        return src;
    }
    VarintU32::get_varint_prefixed_slice(src).unwrap_or(src)
}

#[cfg(test)]
mod tests {
    use crate::db::format::{InternalKeyComparator, LookupKey, ParsedInternalKey, ValueType};
    use crate::iterator::Iterator;
    use crate::mem::memtable::MemTable;
    use crate::util::comparator::BytewiseComparator;
    use std::str;

    fn new_mem_table() -> MemTable<BytewiseComparator> {
        let icmp = InternalKeyComparator::new(BytewiseComparator::default());
        MemTable::new(1 << 32, icmp)
    }

    fn add_test_data_set(memtable: &MemTable<BytewiseComparator>) -> Vec<(&str, &str)> {
        let tests = vec![
            (2, ValueType::Value, "boo", "boo"),
            (4, ValueType::Value, "foo", "val3"),
            (3, ValueType::Deletion, "foo", ""),
            (2, ValueType::Value, "foo", "val2"),
            (1, ValueType::Value, "foo", "val1"),
        ];
        let mut results = vec![];
        for (seq, t, key, value) in tests.clone().drain(..) {
            memtable.add(seq, t, key.as_bytes(), value.as_bytes());
            results.push((key, value));
        }
        results
    }

    #[test]
    fn test_memtable_add_get() {
        let memtable = new_mem_table();
        memtable.add(1, ValueType::Value, b"foo", b"val1");
        memtable.add(2, ValueType::Value, b"foo", b"val2");
        memtable.add(3, ValueType::Deletion, b"foo", b"");
        memtable.add(4, ValueType::Value, b"foo", b"val3");
        memtable.add(2, ValueType::Value, b"boo", b"boo");

        let v = memtable.get(&LookupKey::new(b"null", 10));
        assert!(v.is_none());
        let v = memtable.get(&LookupKey::new(b"foo", 10));
        assert_eq!(b"val3", v.unwrap().unwrap().as_slice());
        let v = memtable.get(&LookupKey::new(b"foo", 0));
        assert!(v.is_none());
        let v = memtable.get(&LookupKey::new(b"foo", 1));
        assert_eq!(b"val1", v.unwrap().unwrap().as_slice());
        let v = memtable.get(&LookupKey::new(b"foo", 3));
        assert!(v.unwrap().is_err());
        let v = memtable.get(&LookupKey::new(b"boo", 3));
        assert_eq!(b"boo", v.unwrap().unwrap().as_slice());
    }

    #[test]
    fn test_memtable_iter() {
        let memtable = new_mem_table();
        let mut iter = memtable.iter();
        assert!(!iter.valid());
        let entries = add_test_data_set(&memtable);
        // Forward scan
        iter.seek_to_first();
        assert!(iter.valid());
        for (key, value) in entries.iter() {
            let k = iter.key();
            let pkey = ParsedInternalKey::decode_from(k).unwrap();
            assert_eq!(
                pkey.as_str(),
                *key,
                "expected key: {:?}, but got {:?}",
                *key,
                pkey.as_str()
            );
            assert_eq!(
                str::from_utf8(iter.value()).unwrap(),
                *value,
                "expected value: {:?}, but got {:?}",
                *value,
                str::from_utf8(iter.value()).unwrap()
            );
            iter.next();
        }
        assert!(!iter.valid());

        // Backward scan
        iter.seek_to_last();
        assert!(iter.valid());
        for (key, value) in entries.iter().rev() {
            let k = iter.key();
            let pkey = ParsedInternalKey::decode_from(k).unwrap();
            assert_eq!(
                pkey.as_str(),
                *key,
                "expected key: {:?}, but got {:?}",
                *key,
                pkey.as_str()
            );
            assert_eq!(
                str::from_utf8(iter.value()).unwrap(),
                *value,
                "expected value: {:?}, but got {:?}",
                *value,
                str::from_utf8(iter.value()).unwrap()
            );
            iter.prev();
        }
        assert!(!iter.valid());
    }
}
