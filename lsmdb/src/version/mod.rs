use crate::cache::table_cache::TableCache;
use crate::db::format::{
    InternalKey, InternalKeyComparator, LookupKey, ParsedInternalKey, ValueType, MAX_KEY_SEQUENCE,
    VALUE_TYPE_FOR_SEEK,
};
use crate::options::{Options, ReadOptions};
use crate::storage::Storage;
use crate::version::version_edit::FileMetaData;
use crate::Result;
use crate::{Comparator, Error, Iterator};
use crossbeam_utils::sync::ShardedLock;
use std::cmp::Ordering as CmpOrdering;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub mod version_edit;

/// A helper for representing the file has been seeked
#[derive(Debug)]
pub struct SeekStats {
    // the file has been seeked
    pub file: Arc<FileMetaData>,
    // the level the 'seek_file' is at
    pub level: usize,
}

pub struct Version<C: Comparator> {
    vnum: usize, // for debug
    options: Arc<Options<C>>,
    icmp: InternalKeyComparator<C>,
    // files per level in this version
    // sorted by the smallest key in FileMetaData
    files: Vec<Vec<Arc<FileMetaData>>>,
    // next file to compact based on seek stats
    file_to_compact: ShardedLock<Option<Arc<FileMetaData>>>,
    file_to_compact_level: AtomicUsize,
    // level that should be compacted next and its compaction score
    // score < 1 means compaction is not strictly needed.
    // These fields are initialized by `finalize`
    compaction_score: f32,
    compaction_level: usize,
}

impl<C: Comparator> fmt::Debug for Version<C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "vnum: {} ", &self.vnum)?;
        for (level, files) in self.files.iter().enumerate() {
            write!(f, "level {}: [ ", level)?;
            for file in files {
                write!(
                    f,
                    "File {}({}): [{:?}..{:?}], ",
                    file.number, file.file_size, file.smallest, file.largest
                )?;
            }
            writeln!(f, " ]")?;
        }
        Ok(())
    }
}

impl<C> Version<C>
where
    C: Comparator + 'static,
{
    pub fn new(options: Arc<Options<C>>, icmp: InternalKeyComparator<C>) -> Self {
        let max_levels = options.max_levels as usize;
        let mut files = Vec::with_capacity(max_levels);
        for _ in 0..max_levels {
            files.push(vec![]);
        }
        Self {
            vnum: 0,
            options,
            icmp,
            files,
            file_to_compact: ShardedLock::new(None),
            file_to_compact_level: AtomicUsize::new(0),
            compaction_score: 0.0,
            compaction_level: 0,
        }
    }

    /// Search the value by the given key in sstables level by level
    pub fn get<S: Storage + Clone + 'static>(
        &self,
        options: ReadOptions,
        key: LookupKey,
        table_cache: &TableCache<S, C>,
    ) -> Result<(Option<Vec<u8>>, Option<SeekStats>)> {
        let ikey = key.internal_key();
        let ukey = key.user_key();
        let ucmp = &self.icmp.user_comparator;
        let mut seek_stats = None;
        let mut files_to_seek = vec![];
        for (level, files) in self.files.iter().enumerate() {
            if files.is_empty() {
                continue;
            }
            if level == 0 {
                for f in files.iter().rev() {
                    if ucmp.compare(ukey, f.smallest.user_key()) != CmpOrdering::Less
                        && ucmp.compare(ukey, f.largest.user_key()) != CmpOrdering::Greater
                    {
                        files_to_seek.push((f, 0));
                    }
                }
            } else {
                let index = find_file(&self.icmp, files, &ikey);
                if index < files.len() {
                    let target = &files[index];
                    if ucmp.compare(ukey, target.smallest.user_key()) != CmpOrdering::Less
                        && level + 1 < self.options.max_levels as usize
                    {
                        files_to_seek.push((target, level));
                    }
                }
            }
        }
        files_to_seek.sort_by(|(a, _), (b, _)| b.number.cmp(&a.number));
        for (file, level) in files_to_seek {
            if seek_stats.is_none() {
                seek_stats = Some(SeekStats {
                    file: file.clone(),
                    level,
                });
            }
            match table_cache.get(
                self.icmp.clone(),
                options,
                &ikey,
                file.number,
                file.file_size,
            )? {
                None => continue,
                Some(block_iter) => {
                    let encoded_key = block_iter.key();
                    let value = block_iter.value();
                    match ParsedInternalKey::decode_from(encoded_key) {
                        None => return Err(Error::Corruption("bad internal key".to_owned())),
                        Some(parsed_key) => {
                            if self
                                .options
                                .comparator
                                .compare(&parsed_key.user_key, key.user_key())
                                == CmpOrdering::Equal
                            {
                                match parsed_key.value_type {
                                    ValueType::Value => {
                                        return Ok((Some(value.to_vec()), seek_stats))
                                    }
                                    ValueType::Deletion => return Ok((None, seek_stats)),
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok((None, seek_stats))
    }

    /// Update seek stats for a sstable file. If it runs out of `allow_seek`,
    /// mark it as a pending compaction file and returns true.
    pub fn update_stats(&self, stats: Option<SeekStats>) -> bool {
        if let Some(ss) = stats {
            let old = ss
                .file
                .allowed_seeks
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |v| {
                    Some(if v > 0 { v - 1 } else { 0 })
                })
                .unwrap();
            let mut file_to_compact = self.file_to_compact.write().unwrap();
            if file_to_compact.is_none() && old == 1 {
                *file_to_compact = Some(ss.file);
                self.file_to_compact_level
                    .store(ss.level, Ordering::Release);
                return true;
            }
        }
        false
    }

    /// Whether the version needs to be compacted
    pub fn needs_compaction(&self) -> bool {
        self.compaction_score > 1.0 || self.file_to_compact.read().unwrap().is_some()
    }

    /// Return a String includes number of files in every level
    pub fn level_summary(&self) -> String {
        let mut s = String::from("files[ ");
        let summary = self.files.iter().fold(String::new(), |mut acc, files| {
            acc.push_str(format!("{} ", files.len()).as_str());
            acc
        });
        s.push_str(summary.as_str());
        s.push(']');
        s
    }

    /// Return the level at which we should place a new memtable compaction
    /// result that covers the range `[smallest_user_key,largest_user_key]`.
    pub fn pick_level_for_memtable_output(
        &self,
        smallest_ukey: &[u8],
        largest_ukey: &[u8],
    ) -> usize {
        let mut level = 0;
        if !self.overlap_in_level(level, Some(smallest_ukey), Some(largest_ukey)) {
            // No overlapping in level 0
            // we might directly push files to next level if there is no overlap in next level
            let smallest_ikey =
                InternalKey::new(smallest_ukey, MAX_KEY_SEQUENCE, VALUE_TYPE_FOR_SEEK);
            let largest_ikey = InternalKey::new(largest_ukey, 0, ValueType::Deletion);
            while level < self.options.max_mem_compact_level {
                // Stops if overlaps at next level
                if self.overlap_in_level(level + 1, Some(smallest_ukey), Some(largest_ukey)) {
                    break;
                }
                if level + 2 < self.options.max_levels as usize {
                    // Check that file does not overlap too many grandparent bytes
                    let overlaps = self.get_overlapping_inputs(
                        level + 2,
                        Some(&smallest_ikey),
                        Some(&largest_ikey),
                    );
                    if total_file_size(&overlaps) > self.options.max_grandparent_overlap_bytes() {
                        break;
                    }
                }
                level += 1;
            }
        }
        level
    }

    /// Returns true iff some file in the specified level overlaps
    /// some part of `[smallest_ukey,largest_ukey]`.
    /// `smallest_ukey` is `None` represents a key smaller than all the DB's keys.
    /// `largest_ukey` is `None` represents a key largest than all the DB's keys.
    pub fn overlap_in_level(
        &self,
        level: usize,
        smallest_ukey: Option<&[u8]>,
        largest_ukey: Option<&[u8]>,
    ) -> bool {
        some_file_overlap_range(
            &self.icmp,
            level > 0,
            &self.files[level],
            smallest_ukey,
            largest_ukey,
        )
    }
}

fn find_file<C: Comparator>(
    icmp: &InternalKeyComparator<C>,
    files: &[Arc<FileMetaData>],
    ikey: &[u8],
) -> usize {
    let mut left = 0;
    let mut right = files.len();
    while left < right {
        let mid = (left + right) >> 1;
        let f = &files[mid];
        if icmp.compare(f.largest.data(), ikey) == CmpOrdering::Less {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    right
}


fn some_file_overlap_range<C: Comparator>(
    icmp: &InternalKeyComparator<C>,
    disjoint: bool,
    files: &[Arc<FileMetaData>],
    smallest_ukey: Option<&[u8]>,
    largest_ukey: Option<&[u8]>,
) -> bool {
    if !disjoint {
        for file in files {
            if key_is_after_file(icmp, file, smallest_ukey)
                || key_is_before_file(icmp, file, largest_ukey)
            {
                // No overlap
                continue;
            } else {
                return true;
            }
        }
        return false;
    }
    // binary search since file ranges are disjoint
    let index = {
        if let Some(s_ukey) = smallest_ukey {
            let smallest_ikey = InternalKey::new(s_ukey, MAX_KEY_SEQUENCE, VALUE_TYPE_FOR_SEEK);
            find_file(icmp, files, smallest_ikey.data())
        } else {
            0
        }
    };
    if index >= files.len() {
        // beginning of range is after all files, so no overlap
        return false;
    }
    // check whether the upper bound is overlapping
    !key_is_before_file(icmp, &files[index], largest_ukey)
}


// used for smallest user key
fn key_is_after_file<C: Comparator>(
    icmp: &InternalKeyComparator<C>,
    file: &Arc<FileMetaData>,
    ukey: Option<&[u8]>,
) -> bool {
    ukey.is_some()
        && icmp
        .user_comparator
        .compare(ukey.unwrap(), file.largest.user_key())
        == CmpOrdering::Greater
}

// used for biggest user key
fn key_is_before_file<C: Comparator>(
    icmp: &InternalKeyComparator<C>,
    file: &Arc<FileMetaData>,
    ukey: Option<&[u8]>,
) -> bool {
    ukey.is_some()
        && icmp
        .user_comparator
        .compare(ukey.unwrap(), file.smallest.user_key())
        == CmpOrdering::Less
}
