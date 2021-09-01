use crate::cache::Cache;
use crate::util::collections::HashMap;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::{mem, ptr};

#[derive(Clone, Copy)]
struct Key<K> {
    k: *const K,
}

impl<K: Hash> Hash for Key<K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe { (*self.k).hash(state) }
    }
}

impl<K: Eq> PartialEq<Self> for Key<K> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { (*self.k).eq(&*other.k) }
    }
}

impl<K: Eq> Eq for Key<K> {}

impl<K> Default for Key<K> {
    fn default() -> Self {
        Self { k: ptr::null() }
    }
}

struct LRUEntry<K, V> {
    key: MaybeUninit<K>,
    value: MaybeUninit<V>,
    prev: *mut LRUEntry<K, V>,
    next: *mut LRUEntry<K, V>,
    charge: usize,
}

impl<K, V> LRUEntry<K, V> {
    fn new(key: K, value: V, charge: usize) -> Self {
        Self {
            key: MaybeUninit::new(key),
            value: MaybeUninit::new(value),
            prev: ptr::null_mut(),
            next: ptr::null_mut(),
            charge,
        }
    }

    fn new_empty() -> Self {
        Self {
            key: MaybeUninit::uninit(),
            value: MaybeUninit::uninit(),
            prev: ptr::null_mut(),
            next: ptr::null_mut(),
            charge: 0,
        }
    }
}

struct LRUInner<K, V> {
    table: HashMap<Key<K>, Box<LRUEntry<K, V>>>,
    // head.next is the newest entry
    head: *mut LRUEntry<K, V>,
    tail: *mut LRUEntry<K, V>,
}

impl<K, V> LRUInner<K, V> {
    fn detach(&mut self, node: *mut LRUEntry<K, V>) {
        unsafe {
            (*(*node).next).prev = (*node).prev;
            (*(*node).prev).next = (*node).next;
        }
    }

    fn attach(&mut self, node: *mut LRUEntry<K, V>) {
        unsafe {
            (*node).next = (*self.head).next;
            (*node).prev = self.head;
            (*self.head).next = node;
            (*(*node).next).prev = node;
        }
    }
}

pub struct LRUCache<K, V: Clone> {
    capacity: usize,
    inner: Arc<Mutex<LRUInner<K, V>>>,
    usage: Arc<AtomicUsize>,
    // Only for tests
    evict_hook: Option<Box<dyn Fn(&K, &V)>>,
}

impl<K, V> LRUCache<K, V>
where
    K: Hash + Eq,
    V: Clone,
{
    pub fn new(capacity: usize) -> Self {
        let inner = LRUInner {
            table: HashMap::default(),
            head: Box::into_raw(Box::new(LRUEntry::new_empty())),
            tail: Box::into_raw(Box::new(LRUEntry::new_empty())),
        };
        unsafe {
            (*inner.head).next = inner.tail;
            (*inner.tail).prev = inner.head;
        }
        Self {
            capacity,
            inner: Arc::new(Mutex::new(inner)),
            usage: Arc::new(AtomicUsize::new(0)),
            evict_hook: None,
        }
    }
}

impl<K, V> Cache<K, V> for LRUCache<K, V>
where
    K: Send + Sync + Hash + Eq + Debug,
    V: Send + Sync + Clone,
{
    fn insert(&self, key: K, mut value: V, charge: usize) -> Option<V> {
        let mut inner = self.inner.lock().unwrap();
        if self.capacity > 0 {
            match inner.table.get_mut(&Key { k: &key as _ }) {
                Some(entry) => {
                    let old_ptr = entry as *mut Box<LRUEntry<K, V>>;
                    unsafe {
                        mem::swap(&mut value, &mut (*(*old_ptr).value.as_mut_ptr()));
                    }
                    let ptr = entry.as_mut() as _;
                    inner.detach(ptr);
                    inner.attach(ptr);
                    if let Some(cb) = &self.evict_hook {
                        cb(&key, &value);
                    }
                    Some(value)
                }
                None => {
                    let mut node = {
                        if self.usage.load(Ordering::Acquire) >= self.capacity {
                            let prev_k = Key {
                                k: unsafe { (*(*inner.tail).prev).key.as_ptr() },
                            };
                            let mut entry = inner.table.remove(&prev_k).unwrap();
                            self.usage.fetch_sub(entry.charge, Ordering::Relaxed);
                            if let Some(cb) = &self.evict_hook {
                                unsafe {
                                    cb(&(*entry.key.as_ptr()), &(*entry.value.as_ptr()));
                                }
                            }
                            unsafe {
                                ptr::drop_in_place(entry.key.as_mut_ptr());
                                ptr::drop_in_place(entry.value.as_mut_ptr());
                            }
                            entry.key = MaybeUninit::new(key);
                            entry.value = MaybeUninit::new(value);
                            inner.detach(entry.as_mut());
                            entry
                        } else {
                            Box::new(LRUEntry::new(key, value, charge))
                        }
                    };
                    self.usage.fetch_add(charge, Ordering::Relaxed);
                    inner.attach(node.as_mut());
                    inner.table.insert(
                        Key {
                            k: node.key.as_ptr(),
                        },
                        node,
                    );
                    None
                }
            }
        } else {
            None
        }
    }

    fn get(&self, key: &K) -> Option<V> {
        let k = Key { k: key as _ };
        let mut inner = self.inner.lock().unwrap();
        if let Some(node) = inner.table.get_mut(&k) {
            let p = node.as_mut() as _;
            inner.detach(p);
            inner.attach(p);
            Some(unsafe { (*(*p).value.as_ptr()).clone() })
        } else {
            None
        }
    }

    fn erase(&self, key: &K) {
        let k = Key { k: key as _ };
        let mut inner = self.inner.lock().unwrap();
        if let Some(mut node) = inner.table.remove(&k) {
            self.usage.fetch_sub(node.charge, Ordering::SeqCst);
            inner.detach(node.as_mut() as _);
            unsafe {
                if let Some(cb) = &self.evict_hook {
                    cb(key, &(*node.value.as_ptr()));
                }
            }
        }
    }

    fn total_charge(&self) -> usize {
        self.usage.load(Ordering::Acquire)
    }
}

impl<K, V: Clone> Drop for LRUCache<K, V> {
    fn drop(&mut self) {
        let mut inner = self.inner.lock().unwrap();
        (*inner).table.values_mut().for_each(|entry| unsafe {
            ptr::drop_in_place(entry.key.as_mut_ptr());
            ptr::drop_in_place(entry.value.as_mut_ptr());
        });
        unsafe {
            let _to_drop_head = *Box::from_raw(inner.head);
            let _to_drop_tail = *Box::from_raw(inner.tail);
        }
    }
}

unsafe impl<K: Send, V: Send + Clone> Send for LRUCache<K, V> {}
unsafe impl<K: Sync, V: Sync + Clone> Sync for LRUCache<K, V> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    const CACHE_SIZE: usize = 100;

    struct CacheTest {
        cache: LRUCache<u32, u32>,
        deleted_kv: Rc<RefCell<Vec<(u32, u32)>>>,
    }

    impl CacheTest {
        fn new(cap: usize) -> Self {
            let deleted_kv = Rc::new(RefCell::new(vec![]));
            let cloned = deleted_kv.clone();
            let mut cache = LRUCache::<u32, u32>::new(cap);
            cache.evict_hook = Some(Box::new(move |k, v| {
                cloned.borrow_mut().push((*k, *v));
            }));
            Self { cache, deleted_kv }
        }

        fn get(&self, key: u32) -> Option<u32> {
            self.cache.get(&key)
        }

        fn insert(&self, key: u32, value: u32) {
            self.cache.insert(key, value, 1);
        }

        fn insert_with_charge(&self, key: u32, value: u32, charge: usize) {
            self.cache.insert(key, value, charge);
        }

        fn erase(&self, key: u32) {
            self.cache.erase(&key);
        }

        fn assert_deleted_kv(&self, index: usize, (key, val): (u32, u32)) {
            assert_eq!((key, val), self.deleted_kv.borrow()[index]);
        }

        fn assert_get(&self, key: u32, want: u32) -> u32 {
            let h = self.cache.get(&key).unwrap();
            assert_eq!(want, h);
            h
        }
    }

    #[test]
    fn test_hit_and_miss() {
        let cache = CacheTest::new(CACHE_SIZE);
        assert_eq!(None, cache.get(100));
        cache.insert(100, 101);
        assert_eq!(Some(101), cache.get(100));
        assert_eq!(None, cache.get(200));
        assert_eq!(None, cache.get(300));

        cache.insert(200, 201);
        assert_eq!(Some(101), cache.get(100));
        assert_eq!(Some(201), cache.get(200));
        assert_eq!(None, cache.get(300));

        cache.insert(100, 102);
        assert_eq!(Some(102), cache.get(100));
        assert_eq!(Some(201), cache.get(200));
        assert_eq!(None, cache.get(300));

        assert_eq!(1, cache.deleted_kv.borrow().len());
        cache.assert_deleted_kv(0, (100, 101));
    }

    #[test]
    fn test_erase() {
        let cache = CacheTest::new(CACHE_SIZE);
        cache.erase(200);
        assert_eq!(0, cache.deleted_kv.borrow().len());

        cache.insert(100, 101);
        cache.insert(200, 201);
        cache.erase(100);

        assert_eq!(None, cache.get(100));
        assert_eq!(Some(201), cache.get(200));
        assert_eq!(1, cache.deleted_kv.borrow().len());
        cache.assert_deleted_kv(0, (100, 101));

        cache.erase(100);
        assert_eq!(None, cache.get(100));
        assert_eq!(Some(201), cache.get(200));
        assert_eq!(1, cache.deleted_kv.borrow().len());
    }

    #[test]
    fn test_entries_are_pinned() {
        let cache = CacheTest::new(CACHE_SIZE);
        cache.insert(100, 101);
        let v1 = cache.assert_get(100, 101);
        assert_eq!(v1, 101);
        cache.insert(100, 102);
        let v2 = cache.assert_get(100, 102);
        assert_eq!(1, cache.deleted_kv.borrow().len());
        cache.assert_deleted_kv(0, (100, 101));
        assert_eq!(v1, 101);
        assert_eq!(v2, 102);

        cache.erase(100);
        assert_eq!(v1, 101);
        assert_eq!(v2, 102);
        assert_eq!(None, cache.get(100));
        assert_eq!(
            vec![(100, 101), (100, 102)],
            cache.deleted_kv.borrow().clone()
        );
    }

    #[test]
    fn test_eviction_policy() {
        let cache = CacheTest::new(CACHE_SIZE);
        cache.insert(100, 101);
        cache.insert(200, 201);
        cache.insert(300, 301);

        // frequently used entry must be kept around
        for i in 0..(CACHE_SIZE + 100) as u32 {
            cache.insert(1000 + i, 2000 + i);
            assert_eq!(Some(2000 + i), cache.get(1000 + i));
            assert_eq!(Some(101), cache.get(100));
        }
        assert_eq!(cache.cache.inner.lock().unwrap().table.len(), CACHE_SIZE);
        assert_eq!(Some(101), cache.get(100));
        assert_eq!(None, cache.get(200));
        assert_eq!(None, cache.get(300));
    }

    #[test]
    fn test_use_exceeds_cache_size() {
        let cache = CacheTest::new(CACHE_SIZE);
        let extra = 100;
        let total = CACHE_SIZE + extra;
        // overfill the cache, keeping handles on all inserted entries
        for i in 0..total as u32 {
            cache.insert(1000 + i, 2000 + i)
        }

        // check that all the entries can be found in the cache
        for i in 0..total as u32 {
            if i < extra as u32 {
                assert_eq!(None, cache.get(1000 + i))
            } else {
                assert_eq!(Some(2000 + i), cache.get(1000 + i))
            }
        }
    }

    #[test]
    fn test_heavy_entries() {
        let cache = CacheTest::new(CACHE_SIZE);
        let light = 1;
        let heavy = 10;
        let mut added = 0;
        let mut index = 0;
        while added < 2 * CACHE_SIZE {
            let weight = if index & 1 == 0 { light } else { heavy };
            cache.insert_with_charge(index, 1000 + index, weight);
            added += weight;
            index += 1;
        }
        let mut cache_weight = 0;
        for i in 0..index {
            let weight = if index & 1 == 0 { light } else { heavy };
            if let Some(val) = cache.get(i) {
                cache_weight += weight;
                assert_eq!(1000 + i, val);
            }
        }
        assert!(cache_weight < CACHE_SIZE);
    }

    #[test]
    fn test_zero_size_cache() {
        let cache = CacheTest::new(0);
        cache.insert(100, 101);
        assert_eq!(None, cache.get(100));
    }
}
