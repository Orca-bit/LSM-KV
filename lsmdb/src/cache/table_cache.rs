use crate::cache::lru::LRUCache;
use crate::cache::Cache;
use crate::db::filename::{generate_filename, FileType};
use crate::options::{Options, ReadOptions};
use crate::sstable::block::BlockIterator;
use crate::sstable::table::{new_table_iterator, Table, TableIterator};
use crate::storage::Storage;
use crate::Comparator;
use crate::Result;
use std::sync::Arc;

pub struct TableCache<S: Storage + Clone, C: Comparator> {
    storage: S,
    db_path: String,
    options: Arc<Options<C>>,
    // the key is the file number
    cache: Arc<dyn Cache<u64, Arc<Table<S::F>>>>,
}

impl<S: Storage + Clone, C: Comparator + 'static> TableCache<S, C> {
    pub fn new(db_path: String, options: Arc<Options<C>>, size: usize, storage: S) -> Self {
        let cache = Arc::new(LRUCache::<u64, Arc<Table<S::F>>>::new(size));
        Self {
            storage,
            db_path,
            options,
            cache,
        }
    }

    /// Try to find the sst file from cache. If not found, try to find the file from storage and insert it into the cache
    pub fn find_table<TC: Comparator>(
        &self,
        cmp: TC,
        file_number: u64,
        file_size: u64,
    ) -> Result<Arc<Table<S::F>>> {
        match self.cache.get(&file_number) {
            Some(file) => Ok(file),
            None => {
                let filename = generate_filename(&self.db_path, FileType::Table, file_number);
                let table_file = self.storage.open(&filename)?;
                let table = Table::open(
                    table_file,
                    file_number,
                    file_size,
                    self.options.clone(),
                    cmp,
                )?;
                let value = Arc::new(table);
                self.cache.insert(file_number, value.clone(), 1);
                Ok(value)
            }
        }
    }

    /// Evict any entry for the specified file number
    pub fn evict(&self, file_number: u64) {
        self.cache.erase(&file_number);
    }

    /// Returns the result of a seek to internal key `key` in specified file
    pub fn get<TC: Comparator>(
        &self,
        cmp: TC,
        options: ReadOptions,
        key: &[u8],
        file_number: u64,
        file_size: u64,
    ) -> Result<Option<BlockIterator<TC>>> {
        let table = self.find_table(cmp.clone(), file_number, file_size)?;
        table.internal_get(options, cmp, key)
    }

    /// Create an iterator for the specified `file_number` (the corresponding
    /// file length must be exactly `file_size` bytes).
    /// The table referenced by returning Iterator will be released after the Iterator is dropped.
    ///
    /// Entry format:
    ///     key: internal key
    ///     value: value of user key
    pub fn new_iter<TC: Comparator>(
        &self,
        cmp: TC,
        options: ReadOptions,
        file_number: u64,
        file_size: u64,
    ) -> Result<TableIterator<TC, S::F>> {
        let table = self.find_table(cmp.clone(), file_number, file_size)?;
        let iter = new_table_iterator(options, table, cmp);
        Ok(iter)
    }
}

impl<S, C> Clone for TableCache<S, C>
where
    S: Storage + Clone,
    C: Comparator,
{
    fn clone(&self) -> Self {
        TableCache {
            storage: self.storage.clone(),
            db_path: self.db_path.clone(),
            options: self.options.clone(),
            cache: self.cache.clone(),
        }
    }
}
