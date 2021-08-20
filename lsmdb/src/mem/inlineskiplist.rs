use crate::iterator::Iterator;
use crate::mem::arena::Arena;
use crate::Comparator;
use crate::Result;
use bytes::Bytes;
use rand::random;
use std::cmp::Ordering as CmpOrdering;
use std::ptr::{null, null_mut, NonNull};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Arc;
use std::{mem, ptr};

const MAX_HEIGHT: usize = 20;
const HEIGHT_INCREASE: u32 = u32::MAX / 3;

#[derive(Debug)]
#[repr(C)]
pub struct Node {
    key: Bytes,
    height: usize,
    next_nodes: [AtomicPtr<Self>; MAX_HEIGHT],
}

impl Node {
    fn new<A: Arena>(key: Bytes, height: usize, arena: &A) -> *mut Self {
        let size =
            mem::size_of::<Self>() - (MAX_HEIGHT - height) * mem::size_of::<AtomicPtr<Self>>();
        let align = mem::align_of::<Self>();
        let p = unsafe { arena.allocate::<Node>(size, align) };
        assert!(!p.is_null());
        unsafe {
            let node = &mut *p;
            ptr::write(&mut node.key, key);
            ptr::write(&mut node.height, height);
            ptr::write_bytes(node.next_nodes.as_mut_ptr(), 0, height);
            p
        }
    }

    #[inline]
    fn get_next(&self, height: usize) -> *mut Node {
        self.next_nodes[height].load(Ordering::SeqCst)
    }

    #[inline]
    fn set_next(&self, height: usize, node: *mut Node) {
        self.next_nodes[height].store(node, Ordering::SeqCst);
    }

    #[inline]
    fn key(&self) -> &[u8] {
        &self.key
    }
}

struct InlineSkipListInner<A: Arena> {
    height: AtomicUsize,
    head: NonNull<Node>,
    arena: A,
    size: AtomicUsize,
}

unsafe impl<A: Arena + Send> Send for InlineSkipListInner<A> {}
unsafe impl<A: Arena + Sync> Sync for InlineSkipListInner<A> {}

impl<A: Arena> Drop for InlineSkipListInner<A> {
    fn drop(&mut self) {
        let mut node = self.head.as_ptr();
        loop {
            let next = unsafe { (&*node).get_next(0) };
            if !next.is_null() {
                unsafe {
                    ptr::drop_in_place(node);
                }
                node = next;
                continue;
            }
            unsafe { ptr::drop_in_place(node) };
            return;
        }
    }
}

#[derive(Clone)]
pub struct InlineSkipList<C, A>
where
    C: Comparator,
    A: Arena + Clone + Send + Sync,
{
    inner: Arc<InlineSkipListInner<A>>,
    comparator: C,
}

impl<C, A> InlineSkipList<C, A>
where
    C: Comparator,
    A: Arena + Clone + Send + Sync,
{
    pub fn new(comparator: C, arena: A) -> Self {
        let head = Node::new(Bytes::new(), MAX_HEIGHT, &arena);
        Self {
            inner: Arc::new(InlineSkipListInner {
                height: AtomicUsize::new(1),
                head: unsafe { NonNull::new_unchecked(head) },
                arena,
                size: AtomicUsize::new(0),
            }),
            comparator,
        }
    }

    fn find_near(&self, key: &[u8], less: bool, allow_equal: bool) -> (*mut Node, bool) {
        let head = self.inner.head.as_ptr();
        let mut ptr = head;
        let mut level = self.get_height() - 1;
        loop {
            unsafe {
                let next_ptr = (*ptr).get_next(level);
                if next_ptr.is_null() {
                    if level > 0 {
                        level -= 1;
                        continue;
                    }
                    if !less || ptr == head {
                        return (null_mut(), false);
                    }
                    return (ptr, false);
                }
                let next = &*next_ptr;
                match self.comparator.compare(key, &next.key) {
                    CmpOrdering::Greater => {
                        ptr = next_ptr;
                        continue;
                    }
                    CmpOrdering::Equal => {
                        if allow_equal {
                            return (next_ptr, true);
                        }
                        if !less {
                            return (next.get_next(0), false);
                        }
                        if level > 0 {
                            level -= 1;
                            continue;
                        }
                        if ptr == head {
                            return (null_mut(), false);
                        }
                        return (ptr, false);
                    }
                    CmpOrdering::Less => {
                        if level > 0 {
                            level -= 1;
                            continue;
                        }
                        if !less {
                            return (next_ptr, false);
                        }
                        if ptr == head {
                            return (null_mut(), false);
                        }
                        return (ptr, false);
                    }
                }
            }
        }
    }

    pub fn put(&self, key: impl Into<Bytes>) {
        let key = key.into();
        self.inner.size.fetch_add(key.len(), Ordering::SeqCst);
        let mut list_height = self.get_height();
        let mut prev = vec![null_mut(); MAX_HEIGHT + 1];
        let mut next = vec![null_mut(); MAX_HEIGHT + 1];
        prev[list_height] = self.inner.head.as_ptr();
        for i in (0..list_height).rev() {
            let (p, n) = self.find_splice_for_level(&key, prev[i + 1], i);
            prev[i] = p;
            next[i] = n;
            assert_ne!(prev[i], next[i]);
        }
        let height = random_height();
        let np = Node::new(key, height, &self.inner.arena);

        while height > list_height {
            match self.inner.height.compare_exchange_weak(
                list_height,
                height,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(h) => list_height = h,
            }
        }

        let node = unsafe { &(*np) };
        for i in 0..height {
            loop {
                if prev[i].is_null() {
                    assert!(i > 1);
                    let (p, n) = self.find_splice_for_level(&node.key, self.inner.head.as_ptr(), i);
                    prev[i] = p;
                    next[i] = n;
                    assert_ne!(prev[i], next[i]);
                }
                unsafe {
                    node.set_next(i, next[i]);
                    match &(*prev[i]).next_nodes[i].compare_exchange(
                        next[i],
                        np,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    ) {
                        Ok(_) => {
                            break;
                        }
                        Err(_) => {
                            let (p, n) = self.find_splice_for_level(&node.key, prev[i], i);
                            if p == n {
                                assert_eq!(i, 0, "Equality can only happen on base level");
                                ptr::drop_in_place(np);
                                return;
                            }
                            prev[i] = p;
                            next[i] = n;
                        }
                    }
                }
            }
        }
    }

    fn get_height(&self) -> usize {
        self.inner.height.load(Ordering::Relaxed)
    }

    fn find_splice_for_level(
        &self,
        key: &[u8],
        mut before: *mut Node,
        height: usize,
    ) -> (*mut Node, *mut Node) {
        loop {
            unsafe {
                let next = (&*before).get_next(height);
                if next.is_null() {
                    return (before, null_mut());
                } else {
                    match self.comparator.compare(key, &(*next).key) {
                        CmpOrdering::Equal => return (next, next),
                        CmpOrdering::Less => return (before, next),
                        CmpOrdering::Greater => {
                            before = next;
                        }
                    }
                }
            }
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.find_last().is_null()
    }

    pub fn len(&self) -> usize {
        let mut node = self.inner.head.as_ptr();
        let mut count = 0;
        loop {
            let next = unsafe { (&*node).get_next(0) };
            if !next.is_null() {
                count += 1;
                node = next;
                continue;
            }
            return count;
        }
    }

    #[inline]
    pub fn total_size(&self) -> usize {
        self.inner.size.load(Ordering::SeqCst) + self.inner.arena.memory_used()
    }

    fn find_last(&self) -> *mut Node {
        let mut ptr = self.inner.head.as_ptr();
        let mut level = self.get_height() - 1;
        loop {
            unsafe {
                let next = (*ptr).get_next(level);
                if next.is_null() {
                    if level > 0 {
                        level -= 1;
                    } else {
                        if ptr == self.inner.head.as_ptr() {
                            return null_mut();
                        }
                        return ptr;
                    }
                } else {
                    ptr = next;
                }
            }
        }
    }
}

fn random_height() -> usize {
    let mut height = 1;
    while height < MAX_HEIGHT && random::<u32>() < HEIGHT_INCREASE {
        height += 1;
    }
    height
}

pub struct InlineSkiplistIterator<C, A>
where
    C: Comparator,
    A: Arena + Clone + Send + Sync,
{
    list: InlineSkipList<C, A>,
    node: *const Node,
}

impl<C, A> InlineSkiplistIterator<C, A>
where
    C: Comparator,
    A: Arena + Clone + Send + Sync,
{
    pub fn new(list: InlineSkipList<C, A>) -> Self {
        Self { list, node: null() }
    }
}

impl<C, A> Iterator for InlineSkiplistIterator<C, A>
where
    C: Comparator,
    A: Arena + Clone + Send + Sync,
{
    #[inline]
    fn valid(&self) -> bool {
        !self.node.is_null()
    }

    fn seek_to_first(&mut self) {
        unsafe { self.node = self.list.inner.head.as_ref().get_next(0) }
    }

    fn seek_to_last(&mut self) {
        self.node = self.list.find_last()
    }

    fn seek(&mut self, target: &[u8]) {
        let (node, _) = self.list.find_near(target, false, true);
        self.node = node;
    }

    fn next(&mut self) {
        assert!(self.valid());
        unsafe {
            self.node = (*self.node).get_next(0);
        }
    }

    fn prev(&mut self) {
        assert!(self.valid());
        let (node, _) = self.list.find_near(self.key(), true, false);
        self.node = node;
    }

    fn key(&self) -> &[u8] {
        assert!(self.valid());
        unsafe { (*self.node).key() }
    }

    fn value(&self) -> &[u8] {
        unimplemented!()
    }

    fn status(&mut self) -> crate::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem::arena::OffsetArena;
    use crate::util::comparator::BytewiseComparator;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    fn new_test_skl() -> InlineSkipList<BytewiseComparator, OffsetArena> {
        InlineSkipList::new(
            BytewiseComparator::default(),
            OffsetArena::new_with_capacity(1 << 20),
        )
    }

    #[test]
    fn test_mem_alloc() {
        let cmp = BytewiseComparator::default();
        let arena = OffsetArena::new_with_capacity(1 << 20);
        let l = InlineSkipList::new(cmp, arena);
        // Node size + align mask
        assert_eq!(mem::size_of::<Node>() + 8, l.inner.arena.memory_used());
    }

    #[test]
    fn test_find_near() {
        let cmp = BytewiseComparator::default();
        let arena = OffsetArena::new_with_capacity(1 << 20);
        let l = InlineSkipList::new(cmp, arena);
        for i in 0..1000 {
            let key = format!("{:05}{:08}", i * 10 + 5, 0);
            l.put(key);
        }
        let cases = vec![
            ("00001", false, false, Some("00005")),
            ("00001", false, true, Some("00005")),
            ("00001", true, false, None),
            ("00001", true, true, None),
            ("00005", false, false, Some("00015")),
            ("00005", false, true, Some("00005")),
            ("00005", true, false, None),
            ("00005", true, true, Some("00005")),
            ("05555", false, false, Some("05565")),
            ("05555", false, true, Some("05555")),
            ("05555", true, false, Some("05545")),
            ("05555", true, true, Some("05555")),
            ("05558", false, false, Some("05565")),
            ("05558", false, true, Some("05565")),
            ("05558", true, false, Some("05555")),
            ("05558", true, true, Some("05555")),
            ("09995", false, false, None),
            ("09995", false, true, Some("09995")),
            ("09995", true, false, Some("09985")),
            ("09995", true, true, Some("09995")),
            ("59995", false, false, None),
            ("59995", false, true, None),
            ("59995", true, false, Some("09995")),
            ("59995", true, true, Some("09995")),
        ];
        for (i, (key, less, allow_equal, exp)) in cases.into_iter().enumerate() {
            let seek_key = format!("{}{:08}", key, 0);
            let (res, found) = l.find_near(seek_key.as_bytes(), less, allow_equal);
            if exp.is_none() {
                assert!(!found, "{}", i);
                continue;
            }
            let e = format!("{}{:08}", exp.unwrap(), 0);
            assert_eq!(&unsafe { &*res }.key, e.as_bytes(), "{}", i);
        }
    }

    #[test]
    fn test_empty() {
        let key = b"aaa";
        let skl = new_test_skl();
        for less in &[false, true] {
            for allow_equal in &[false, true] {
                let (node, found) = skl.find_near(key, *less, *allow_equal);
                assert!(node.is_null());
                assert!(!found);
            }
        }
        let mut iter = InlineSkiplistIterator::new(skl.clone());
        assert!(!iter.valid());
        iter.seek_to_first();
        assert!(!iter.valid());
        iter.seek_to_last();
        assert!(!iter.valid());
        iter.seek(key);
        assert!(!iter.valid());
    }

    #[test]
    fn test_basic() {
        let c = BytewiseComparator::default();
        let arena = OffsetArena::new_with_capacity(1 << 20);
        let list = InlineSkipList::new(c, arena);
        let table = vec!["key1", "key2", "key3", "key4", "key5"];

        for key in table.clone() {
            list.put(key.as_bytes());
        }
        assert_eq!(list.len(), 5);
        assert!(!list.is_empty());
        let mut iter = InlineSkiplistIterator::new(list);
        for key in &table {
            iter.seek(key.as_bytes());
            assert_eq!(iter.key(), key.as_bytes());
        }
        for key in table.iter().rev() {
            assert_eq!(iter.key(), key.as_bytes());
            iter.prev();
        }
        assert!(!iter.valid());
        iter.seek_to_first();
        for key in table.iter() {
            assert_eq!(iter.key(), key.as_bytes());
            iter.next();
        }
        assert!(!iter.valid());
        iter.seek_to_first();
        assert_eq!(iter.key(), table.first().unwrap().as_bytes());
        iter.seek_to_last();
        assert_eq!(iter.key(), table.last().unwrap().as_bytes());
    }

    fn test_concurrent_basic(n: usize, cap: usize, key_len: usize) {
        let cmp = BytewiseComparator::default();
        let arena = OffsetArena::new_with_capacity(cap);
        let skl = InlineSkipList::new(cmp, arena);
        let keys: Vec<_> = (0..n)
            .map(|i| format!("{1:00$}", key_len, i).to_owned())
            .collect();
        let (tx, rx) = mpsc::channel();
        for key in keys.clone() {
            let tx = tx.clone();
            let l = skl.clone();
            thread::Builder::new()
                .name("write thread".to_owned())
                .spawn(move || {
                    l.put(key);
                    tx.send(()).unwrap();
                })
                .unwrap();
        }
        for _ in 0..n {
            rx.recv_timeout(Duration::from_secs(3)).unwrap();
        }
        for key in keys {
            let tx = tx.clone();
            let l = skl.clone();
            thread::Builder::new()
                .name("read thread".to_owned())
                .spawn(move || {
                    let mut iter = InlineSkiplistIterator::new(l);
                    iter.seek(key.as_bytes());
                    assert_eq!(iter.key(), key.as_bytes());
                    tx.send(()).unwrap();
                })
                .unwrap();
        }
        for _ in 0..n {
            rx.recv_timeout(Duration::from_secs(3)).unwrap();
        }
        assert_eq!(skl.len(), n);
    }

    #[test]
    fn test_concurrent_basic_small_value() {
        test_concurrent_basic(1000, 1 << 20, 5);
    }
    #[test]
    fn test_concurrent_basic_big_value() {
        test_concurrent_basic(100, 120 << 20, 10);
    }
}
