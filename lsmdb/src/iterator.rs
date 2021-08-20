use crate::util::comparator::Comparator;
use crate::{Error, Result};

pub trait Iterator {
    /// true if position at a existed entry
    /// else false
    fn valid(&self) -> bool;
    /// position at the first entry
    fn seek_to_first(&mut self);
    /// position at the last entry
    fn seek_to_last(&mut self);
    /// seek a existed entry
    fn seek(&mut self, target: &[u8]);
    /// move to next entry
    fn next(&mut self);
    /// move to previous entry
    fn prev(&mut self);
    /// the key of current entry
    fn key(&self) -> &[u8];
    /// the value of current entry
    fn value(&self) -> &[u8];
    /// check the status
    fn status(&mut self) -> Result<()>;
}
