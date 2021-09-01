use crate::util::collections::HashMap;
use std::io::Cursor;
use std::path::MAIN_SEPARATOR;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, RwLock};

#[derive(Clone)]
pub struct MemStorage {
    inner: Arc<RwLock<HashMap<String, Node>>>,

    /// sstable/log `flush()` calls are blocked.
    pub delay_data_sync: Arc<AtomicBool>,

    /// sstable/log `flush()` calls return an error
    pub data_sync_error: Arc<AtomicBool>,

    /// Simulate no-space errors
    pub no_space: Arc<AtomicBool>,

    /// Simulate non-writable file system
    pub non_writable: Arc<AtomicBool>,

    /// Force sync of manifest files to fail
    pub manifest_sync_error: Arc<AtomicBool>,

    /// Force write to manifest files to fail
    pub manifest_write_error: Arc<AtomicBool>,

    /// Whether enable to record the count of random reads to files
    pub count_random_reads: bool,

    pub random_read_counter: Arc<AtomicUsize>,
}

impl Default for MemStorage {
    fn default() -> Self {
        let mut map = HashMap::default();
        map.insert(MAIN_SEPARATOR.to_string(), Node::Dir);
        let inner = Arc::new(RwLock::new(map));
        Self {
            inner,
            delay_data_sync: Arc::new(AtomicBool::new(false)),
            data_sync_error: Arc::new(AtomicBool::new(false)),
            no_space: Arc::new(AtomicBool::new(false)),
            non_writable: Arc::new(AtomicBool::new(false)),
            manifest_sync_error: Arc::new(AtomicBool::new(false)),
            manifest_write_error: Arc::new(AtomicBool::new(false)),
            count_random_reads: false,
            random_read_counter: Arc::new(AtomicUsize::new(0)),
        }
    }
}

enum Node {
    File(FileNode),
    Dir,
}

pub struct FileNode {
    name: String,
    delay_data_sync: Arc<AtomicBool>,
    data_sync_error: Arc<AtomicBool>,
    no_space: Arc<AtomicBool>,

    manifest_sync_error: Arc<AtomicBool>,
    manifest_write_error: Arc<AtomicBool>,

    count_random_reads: Arc<AtomicBool>,
    random_read_counter: Arc<AtomicUsize>,

    inner: Arc<RwLock<InmemFile>>,
}

struct InmemFile {
    lock: AtomicBool,
    contents: Cursor<Vec<u8>>,
}
