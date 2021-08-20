use std::sync::Arc;

const MIN_SNAPSHOT: u64 = 0;

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Copy, Clone)]
pub struct Snapshot {
    sequence_id: u64,
}

impl Snapshot {
    #[inline]
    pub fn sequence(self) -> u64 {
        self.sequence_id
    }
}

impl From<u64> for Snapshot {
    fn from(sequence_id: u64) -> Self {
        Self { sequence_id }
    }
}

pub struct SnapshotList {
    first: Arc<Snapshot>,
    snapshots: Vec<Arc<Snapshot>>,
}

impl Default for SnapshotList {
    fn default() -> Self {
        let first = Arc::new(MIN_SNAPSHOT.into());
        Self {
            first,
            snapshots: Vec::new(),
        }
    }
}

impl SnapshotList {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    #[inline]
    pub(crate) fn oldest(&self) -> Arc<Snapshot> {
        if self.is_empty() {
            self.first.clone()
        } else {
            self.snapshots.first().unwrap().clone()
        }
    }

    #[inline]
    fn newest(&self) -> Arc<Snapshot> {
        if self.is_empty() {
            self.first.clone()
        } else {
            self.snapshots.last().unwrap().clone()
        }
    }

    #[inline]
    fn last_seq(&self) -> u64 {
        self.snapshots
            .last()
            .map_or(self.first.sequence(), |last| last.sequence())
    }

    pub fn acquire(&mut self, sequence_id: u64) -> Arc<Snapshot> {
        let last_seq = self.last_seq();
        assert!(
            sequence_id >= last_seq,
            "[snapshot] the sequence_id must monotonically increase: [new: {}], [last: {}]",
            sequence_id,
            last_seq
        );
        if sequence_id == last_seq {
            self.newest()
        } else {
            let new_snapshot = Arc::new(Snapshot { sequence_id });
            self.snapshots.push(new_snapshot.clone());
            new_snapshot
        }
    }

    #[inline]
    pub fn gc(&mut self) {
        self.snapshots
            .retain(|snapshot| Arc::strong_count(snapshot) > 1)
    }

    #[inline]
    pub fn release(&mut self, snapshot: Arc<Snapshot>) -> bool {
        match self.snapshots.as_slice().binary_search(&snapshot) {
            Ok(i) => {
                self.snapshots.remove(i);
                true
            }
            Err(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_empty() {
        let mut s = SnapshotList::default();
        assert!(s.is_empty());
        assert_eq!(MIN_SNAPSHOT, s.last_seq());
        assert_eq!(MIN_SNAPSHOT, s.acquire(MIN_SNAPSHOT).sequence());
    }

    #[test]
    fn test_oldest() {
        let mut s = SnapshotList::default();
        assert_eq!(MIN_SNAPSHOT, s.oldest().sequence());
        for i in vec![1, 1, 2, 3] {
            s.acquire(i);
        }
    }

    #[test]
    fn test_gc() {
        let mut s = SnapshotList::default();
        s.acquire(1);
        let s2 = s.acquire(2);
        s.acquire(3);
        s.gc();
        assert_eq!(1, s.snapshots.len());
        assert_eq!(s2.sequence(), s.snapshots.pop().unwrap().sequence());
    }

    #[test]
    fn test_append_new_snapshot() {
        let mut s = SnapshotList::default();
        for i in vec![1, 1, 2, 3] {
            let s = s.acquire(i);
            assert_eq!(s.sequence(), i);
        }
        assert_eq!(1, s.oldest().sequence());
        assert_eq!(3, s.newest().sequence());
    }

    #[test]
    fn test_release() {
        let mut s = SnapshotList::default();
        for i in vec![1, 1, 2, 3] {
            s.acquire(i);
        }
        assert!(s.release(Arc::new(Snapshot { sequence_id: 2 })));
        assert_eq!(
            vec![1, 3],
            s.snapshots
                .into_iter()
                .map(|s| s.sequence())
                .collect::<Vec<_>>()
        );
    }
}
